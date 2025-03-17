# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch
import torch.nn as nn
from diffusers.models import attention_processor

T = torch.Tensor

# Copied from diffusers.loaders.unet
def _convert_ip_adapter_image_proj_to_diffusers(self, state_dict):
    updated_state_dict = {}
    image_projection = None

    if "proj.weight" in state_dict:
        # IP-Adapter
        num_image_text_embeds = 4
        clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
        cross_attention_dim = state_dict["proj.weight"].shape[0] // 4

        image_projection = ImageProjection(
            cross_attention_dim=cross_attention_dim,
            image_embed_dim=clip_embeddings_dim,
            num_image_text_embeds=num_image_text_embeds,
        )

        for key, value in state_dict.items():
            diffusers_name = key.replace("proj", "image_embeds")
            updated_state_dict[diffusers_name] = value

    elif "proj.3.weight" in state_dict:
        # IP-Adapter Full
        clip_embeddings_dim = state_dict["proj.0.weight"].shape[0]
        cross_attention_dim = state_dict["proj.3.weight"].shape[0]

        image_projection = MLPProjection(
            cross_attention_dim=cross_attention_dim, image_embed_dim=clip_embeddings_dim
        )

        for key, value in state_dict.items():
            diffusers_name = key.replace("proj.0", "ff.net.0.proj")
            diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
            diffusers_name = diffusers_name.replace("proj.3", "norm")
            updated_state_dict[diffusers_name] = value

    else:
        # IP-Adapter Plus
        num_image_text_embeds = state_dict["latents"].shape[1]
        embed_dims = state_dict["proj_in.weight"].shape[1]
        output_dims = state_dict["proj_out.weight"].shape[0]
        hidden_dims = state_dict["latents"].shape[2]
        heads = state_dict["layers.0.0.to_q.weight"].shape[0] // 64

        image_projection = Resampler(
            embed_dims=embed_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            heads=heads,
            num_queries=num_image_text_embeds,
        )

        for key, value in state_dict.items():
            diffusers_name = key.replace("0.to", "2.to")
            diffusers_name = diffusers_name.replace("1.0.weight", "3.0.weight")
            diffusers_name = diffusers_name.replace("1.0.bias", "3.0.bias")
            diffusers_name = diffusers_name.replace("1.1.weight", "3.1.net.0.proj.weight")
            diffusers_name = diffusers_name.replace("1.3.weight", "3.1.net.2.weight")

            if "norm1" in diffusers_name:
                updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
            elif "norm2" in diffusers_name:
                updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
            elif "to_kv" in diffusers_name:
                v_chunk = value.chunk(2, dim=0)
                updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
            elif "to_out" in diffusers_name:
                updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
            else:
                updated_state_dict[diffusers_name] = value

    image_projection.load_state_dict(updated_state_dict)
    return image_projection


# Copied from diffusers.loaders.unet
def _load_ip_adapter_weights(self, state_dict):
    from ..models.attention_processor import (
        AttnProcessor,
        AttnProcessor2_0,
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0,
    )

    if "proj.weight" in state_dict["image_proj"]:
        # IP-Adapter
        num_image_text_embeds = 4
    elif "proj.3.weight" in state_dict["image_proj"]:
        # IP-Adapter Full Face
        num_image_text_embeds = 257  # 256 CLIP tokens + 1 CLS token
    else:
        # IP-Adapter Plus
        num_image_text_embeds = state_dict["image_proj"]["latents"].shape[1]

    # Set encoder_hid_proj after loading ip_adapter weights,
    # because `Resampler` also has `attn_processors`.
    self.encoder_hid_proj = None

    # set ip-adapter cross-attention processors & load state_dict
    attn_procs = {}
    key_id = 1
    for name in self.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = self.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(self.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = self.config.block_out_channels[block_id]
        if cross_attention_dim is None or "motion_modules" in name:
            attn_processor_class = (
                AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
            )
            attn_procs[name] = attn_processor_class()
        else:
            attn_processor_class = (
                IPAdapterAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else IPAdapterAttnProcessor
            )
            attn_procs[name] = attn_processor_class(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=1.0,
                num_tokens=num_image_text_embeds,
            ).to(dtype=self.dtype, device=self.device)

            value_dict = {}
            for k, w in attn_procs[name].state_dict().items():
                value_dict.update({f"{k}": state_dict["ip_adapter"][f"{key_id}.{k}"]})

            attn_procs[name].load_state_dict(value_dict)
            key_id += 2

    self.set_attn_processor(attn_procs)

    # convert IP-Adapter Image Projection layers to diffusers
    image_projection = self._convert_ip_adapter_image_proj_to_diffusers(state_dict["image_proj"])

    self.encoder_hid_proj = image_projection.to(device=self.device, dtype=self.dtype)
    self.config.encoder_hid_dim_type = "ip_image_proj"

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


 ## for controlnet
class CNAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, num_tokens=4):
        self.num_tokens = num_tokens

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, skip=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.skip = skip

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if not self.skip:
            # for ip-adapter

            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            self.attn_map = ip_attention_probs
            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

            hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class DefaultAttentionProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        self.processor = attention_processor.AttnProcessor()

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask)

