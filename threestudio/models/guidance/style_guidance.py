import random
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionDepth2ImgPipeline,
    UNet2DConditionModel,
    ControlNetModel,
    DDIMScheduler
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.embeddings import TimestepEmbedding,ImageProjection
from diffusers.utils.import_utils import is_xformers_available
from threestudio.utils.ops import perpendicular_component

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from controlnet_aux import NormalBaeDetector, CannyDetector, PidiNetDetector, HEDdetector

from diffusers.utils import BaseOutput

from .attn_processor import IPAttnProcessor,ImageProjModel,DefaultAttentionProcessor,CNAttnProcessor
#region help function
class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->DDIM
class DDIMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None

# Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.step
def ddim_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    delta_timestep: int = None,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise: Optional[torch.FloatTensor] = None,
    return_dict: bool = True,
    **kwargs
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        eta (`float`):
            The weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`, defaults to `False`):
            If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
            because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
            clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
            `use_clipped_model_output` has no effect.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        variance_noise (`torch.FloatTensor`):
            Alternative to generating noise with `generator` by directly providing the noise for the variance
            itself. Useful for methods such as [`CycleDiffusion`].
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    if delta_timestep is None:
        # 1. get previous step value (=t+1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    else:
        prev_timestep = timestep - delta_timestep



    timestep=timestep.to('cpu')
    prev_timestep = prev_timestep.to('cpu')
    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t
    alpha_prod_t=alpha_prod_t.to('cuda:0')
    alpha_prod_t_prev=alpha_prod_t_prev.to('cuda:0')
    beta_prod_t=beta_prod_t.to('cuda:0')


    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # if prev_timestep < timestep:
    # else:
    #     variance = abs(self._get_variance(prev_timestep, timestep))

    variance = abs(self._get_variance(timestep, prev_timestep))

    std_dev_t = (eta * variance).to('cuda:0')
    std_dev_t = min((1 - alpha_prod_t_prev) / 2, std_dev_t) ** 0.5

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if eta > 0:
        if variance_noise is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                " `variance_noise` stays `None`."
            )

        if variance_noise is None:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        variance = std_dev_t * variance_noise

        prev_sample = prev_sample + variance
    
    prev_sample = torch.nan_to_num(prev_sample)

    if not return_dict:
        return (prev_sample,)

    return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

#endregion

@threestudio.register("style-guidance")
class StableDiffusionStyleGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        width:int=512
        height:int=512
        cache_dir: Optional[str] = "../../model/SD"
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        half_precision_weights: bool = True

        # set up for potential control net
        use_controlnet: bool = False
        use_depthdiffusion: bool = False

        condition_scale: float = 1.5
        control_anneal_start_step: Optional[int] = None
        control_anneal_end_scale: Optional[float] = None
        control_types: List = field(default_factory=lambda: ['depth', 'canny'])  
        condition_scales: List = field(default_factory=lambda: [1.0, 1.0])
        condition_scales_anneal: List = field(default_factory=lambda: [1.0, 1.0])
        p2p_condition_type: str = 'p2p'
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_range : float = 0.98


        cond_scale: float = 1
        uncond_scale: float = 0
        null_scale: float = -1
        noise_scale: float = 0
        perpneg_scale: float = 0.0

        view_dependent_prompting: bool = True

        grad_clip_val: Optional[float] = None
        grad_normalize: Optional[
            bool
        ] = False 
        
        grad_scale: float = 0.1
        guidance_scale: float =7.5
        annealing_intervals: bool = True
        delta_t: int = 50
        delta_t_start: int = 100
        xs_delta_t: int = 200
        xs_inv_steps: int = 5
        xs_eta: float = 0.0
        denoise_guidance_scale: float = 1.0

        use_ip_adapter: bool = True
        style_guidance: float = 7.5
        ref_img_path: str = ""
        ref_content_prompt: str= ""
        use_negsub: bool = True
        use_layer_injection: bool = True

    cfg: Config
    #region ipadapter
    def set_ip_adapter_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def set_ips_layer(self, unet,controlnet,num_tokens=4,
        target_blocks=['down_blocks.1.attentions.0','up_blocks',"down_blocks.2",'mid_block.attentions.0'],
        ):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = DefaultAttentionProcessor()
            else:
                selected = False
                for block_name in target_blocks:
                    if block_name in name:
                        selected = True
                        break
                if selected:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=num_tokens,
                    ).to(self.device, dtype=torch.float16)
                else:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=num_tokens,
                        skip=True
                    ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if self.use_controlnet:
            if not isinstance(controlnet, ControlNetModel):
                for controlnet in controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=num_tokens))
            else:
                controlnet.set_attn_processor(CNAttnProcessor(num_tokens=num_tokens))

        
    def load_ip_adapter(self,pretrained_model_name_or_path_or_dict=None,weight_name=None,cache_dir=None,subfolder=None):
        from diffusers.utils import _get_model_file
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                subfolder=subfolder,
                local_files_only = False,
                force_download = None,
                proxies=None,
                resume_download=None,
                token=None,
                user_agent=None,
                revision=None
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

        image_projection = self.pipe.unet._convert_ip_adapter_image_proj_to_diffusers(state_dict["image_proj"])
        self.pipe.unet.encoder_hid_proj = image_projection.to(device= self.pipe.unet.device, dtype= self.pipe.unet.dtype)
        self.pipe.unet.config.encoder_hid_dim_type = "ip_image_proj"
    #endregion
    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")
        self.use_controlnet = self.cfg.use_controlnet

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        if self.use_controlnet:
            self.start_condition_scale = self.cfg.condition_scale
            multi_controlnet_processor = []
            for i, control_type in enumerate(self.cfg.control_types):
                if control_type == 'normal':
                    controlnet_name_or_path: str = "lllyasviel/control_v11p_sd15_normalbae"
                    self.preprocessor_normal = NormalBaeDetector.from_pretrained(
                        "lllyasviel/Annotators",
                        cache_dir=self.cfg.cache_dir,
                    )
                    self.preprocessor_normal.model.to(self.device)
                elif control_type == 'canny':
                    controlnet_name_or_path: str = "lllyasviel/control_v11p_sd15_canny"
                    self.preprocessor_canny = CannyDetector()

                elif control_type == 'self-normal':
                    controlnet_name_or_path: str = "lllyasviel/control_v11p_sd15_normalbae"

                elif control_type == 'hed':
                    controlnet_name_or_path: str = "lllyasviel/control_v11p_sd15_scribble"
                    self.preprocessor_hed = HEDdetector.from_pretrained(
                        'lllyasviel/Annotators',
                        cache_dir=self.cfg.cache_dir,
                        )
                elif control_type == 'p2p':
                    controlnet_name_or_path: str = "lllyasviel/control_v11e_sd15_ip2p"
                elif control_type == 'depth':
                    controlnet_name_or_path: str = "lllyasviel/control_v11f1p_sd15_depth"
                print(control_type)
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_name_or_path,
                    torch_dtype=self.weights_dtype,
                    cache_dir=self.cfg.cache_dir,
                )

                multi_controlnet_processor.append(controlnet)

            pipe_kwargs = {
                #"tokenizer": None,
                "safety_checker": None,
                "feature_extractor": None,
                "controlnet": multi_controlnet_processor,
                "requires_safety_checker": False,
                "torch_dtype": self.weights_dtype,
                "cache_dir": self.cfg.cache_dir,
            }

    
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                **pipe_kwargs)

            self.pipe.to(self.device)
            #self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        
        elif self.cfg.use_depthdiffusion:
            pipe_kwargs = {
                "tokenizer": None,
                "safety_checker": None,
                "feature_extractor": None,
                "requires_safety_checker": False,
                "torch_dtype": self.weights_dtype,
                "cache_dir": self.cfg.cache_dir,
            }
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                **pipe_kwargs,
            ).to(self.device)

        else:
            pipe_kwargs = {
                "tokenizer": None,
                "safety_checker": None,
                "feature_extractor": None,
                "requires_safety_checker": False,
                "torch_dtype": self.weights_dtype,
                "cache_dir": self.cfg.cache_dir,
            }
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                **pipe_kwargs,
            ).to(self.device)

        if self.cfg.use_ip_adapter:
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

            if self.cfg.use_layer_injection:
                self.set_ips_layer(unet=self.pipe.unet,controlnet=self.pipe.controlnet if self.cfg.use_controlnet else None)
            else:
                self.set_ips_layer(unet=self.pipe.unet,controlnet=self.pipe.controlnet if self.cfg.use_controlnet else None
                    ,target_blocks=['block'])
            
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained('h94/IP-Adapter',subfolder="models/image_encoder",cache_dir =self.cfg.cache_dir).to(
                self.device, dtype=torch.float16
            )
            self.clip_image_processor = CLIPImageProcessor()
            self.image_proj_model = ImageProjModel(
                    cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                    clip_embeddings_dim=self.image_encoder.config.projection_dim,
                    clip_extra_context_tokens=4,
                ).to(self.device, dtype=torch.float16)

            self.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin",cache_dir =self.cfg.cache_dir)
        
        self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                cache_dir=self.cfg.cache_dir,
                )
        self.sche_func = ddim_step
        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        if self.use_controlnet:
            self.controlnet = self.pipe.controlnet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)


        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.scheduler.set_timesteps(self.num_train_timesteps, device = self.device)
        self.set_min_max_steps()
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.precision_t = self.weights_dtype

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_ip_adapter:
            self.set_ip_adapter_scale(1.0)

            from diffusers.utils import load_image
            ip_adapter_image = load_image(self.cfg.ref_img_path)
            if ip_adapter_image is not None:

                clip_image = self.clip_image_processor(images=[ip_adapter_image], return_tensors="pt").pixel_values

                image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
                negative_image_embeds = torch.zeros_like(image_embeds)

                if self.cfg.use_negsub:
                    from transformers import CLIPTextModelWithProjection, CLIPTokenizer,logging
                    logging.set_verbosity_error()
                    text_encoder = CLIPTextModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K',cache_dir=self.cfg.cache_dir,use_safetensors=False).to(self.device, 
                            dtype=self.weights_dtype)
                    tokenizer = CLIPTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K',cache_dir=self.cfg.cache_dir)

                    tokens = tokenizer([self.cfg.ref_content_prompt], return_tensors='pt').to(text_encoder.device)
                    neg_content_emb = text_encoder(**tokens).text_embeds

                    image_embeds = image_embeds - image_embeds @ neg_content_emb.transpose(0,1) * neg_content_emb/(neg_content_emb @ neg_content_emb.transpose(0,1))

                self.image_embeddings = torch.cat([image_embeds,negative_image_embeds])

        threestudio.info(f"Loaded Stable Diffusion!")

    #region forward
    def multi_control_forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            controlnet_cond: List[torch.tensor],
            conditioning_scale: List[float],
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.controlnet.nets)):
            down_samples, mid_sample = controlnet(
                sample.to(self.weights_dtype),
                timestep.to(self.weights_dtype),
                encoder_hidden_states.to(self.weights_dtype),
                image.to(self.weights_dtype),
                scale,
                class_labels,
                timestep_cond,
                attention_mask,
                cross_attention_kwargs,
                return_dict=False,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        condition_scale: float,
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Float[Tensor, "..."]] = None,
        mid_block_additional_residual: Optional[Float[Tensor, "..."]] = None
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    #endregion
    
    def compute_without_perpneg(
        self,
        condition_scales,
        text_embeddings,
        latents_noisy,
        depth_latent,
        t, 
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        image_cond: Float[Tensor, "B 3 512 512"],
    ):


        added_cond_kwargs = (
            {"image_embeds": torch.cat([self.image_embeddings[1].unsqueeze(0),self.image_embeddings[0].unsqueeze(0)])}
            if self.cfg.use_ip_adapter
            else None
        )
        with torch.no_grad():
            if (not self.cfg.use_depthdiffusion) or self.cfg.use_controlnet:      
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            else:
                latents_noisy=torch.cat((latents_noisy,depth_latent), dim=1)
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            if not all(scale == 0 for scale in condition_scales):
                down_block_res_samples, mid_block_res_sample = self.multi_control_forward(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=image_cond,
                    conditioning_scale=condition_scales
                )
                with self.disable_unet_class_embedding(self.unet) as unet:
                    noise_pred = self.forward_unet(
                        unet,
                        latent_model_input,
                        torch.cat([t] * 2),
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=added_cond_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample
                    )
            else:
                with self.disable_unet_class_embedding(self.unet) as unet:
                    noise_pred = self.forward_unet(
                        unet,
                        latent_model_input,
                        torch.cat([t] * 2),
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=added_cond_kwargs,
                    )

        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)

        return noise_pred_text, noise_pred_uncond

    #region add_noise_with_cfg
    def add_noise_with_cfg(self, latents, noise,depth_latent,
                           ind_t, ind_prev_t, 
                           text_embeddings=None, cfg=1.0, 
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):
        
        text_embeddings = text_embeddings.to(self.precision_t)
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []
        

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            if self.cfg.use_depthdiffusion:
                cur_noisy_lat_ = torch.cat((cur_noisy_lat_,depth_latent), dim=1).to(self.precision_t)
            latent_model_input = cur_noisy_lat_
            if cfg > 1.0:
                added_cond_kwargs = (
                    {"image_embeds": torch.stack([self.image_embeddings[-1]] * 2)}
                    if self.cfg.use_ip_adapter
                    else None
                )
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

                unet_output = unet(latent_model_input, timestep_model_input, 
                            encoder_hidden_states=text_embeddings,added_cond_kwargs=added_cond_kwargs).sample
            
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                added_cond_kwargs = (
                    {"image_embeds": self.image_embeddings[-1].unsqueeze(0)}
                    if self.cfg.use_ip_adapter
                    else None
                )
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=uncond_text_embedding,added_cond_kwargs=added_cond_kwargs).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t
            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]
    #endregion

    def compute_grad_sds(
        self,
        prompt_utils,
        condition_scales,
        latents: Float[Tensor, "B 4 64 64"],
        depth_latent: Float[Tensor, "B 1 64 64"],
        image_cond: Float[Tensor, "B 3 512 512"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        warm_up_rate = 0
    ):

        text_embeddings = prompt_utils.get_text_embeddings(
                elevation,
                azimuth,
                camera_distances,
                self.cfg.view_dependent_prompting,
                return_null_text_embeddings=False,
            )

        inverse_text_embeddings = prompt_utils.get_inverse_text_embeddings(
                elevation,
                azimuth,
                camera_distances,
                self.cfg.view_dependent_prompting,
            )

        B = latents.shape[0]

        if self.cfg.annealing_intervals:
            current_delta_t =  int(self.cfg.delta_t + np.ceil((warm_up_rate)*(self.cfg.delta_t_start - self.cfg.delta_t)))
        else:
            current_delta_t = self.cfg.delta_t


        ind_t = torch.randint(
            self.min_step,
            self.max_step + int(self.warmup_step*warm_up_rate),
            [B],
            dtype=torch.long,
            device=self.device,
        )
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t)*0)

        
        # random timestamp
        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]
        xs_delta_t = self.cfg.xs_delta_t if self.cfg.xs_delta_t is not None else current_delta_t
        xs_inv_steps = self.cfg.xs_inv_steps if self.cfg.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
        starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

        input_latents = latents

        noise = torch.randn_like(input_latents)
        with torch.no_grad():

            _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(input_latents, noise, depth_latent, ind_prev_t, starting_ind, inverse_text_embeddings, 
                                                                                self.cfg.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=self.cfg.xs_eta)
            _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, depth_latent, ind_t, ind_prev_t, inverse_text_embeddings, 
                                                                            self.cfg.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True)    
            pred_scores = pred_scores_xs + pred_scores_xt

            noise = pred_scores[0][1]

            ipa_noise_pred_ref_text, ipa_noise_pred_text= self.compute_without_perpneg(
                    condition_scales, torch.cat([prompt_utils.ref_text_embeddings,text_embeddings[0].unsqueeze(0)]), latents_noisy,depth_latent, t, elevation, azimuth, camera_distances, image_cond
                )
            
            self.set_ip_adapter_scale(0.0)
            ism_noise_pred_text, ism_noise_pred_uncond = self.compute_without_perpneg(
                condition_scales, text_embeddings, latents_noisy,depth_latent, t, elevation, azimuth, camera_distances, image_cond
            )
            self.set_ip_adapter_scale(1.0)
            
            latents_noisy = torch.cat([latents_noisy,latents_noisy])
            ipa_pred_noise = ism_noise_pred_uncond + self.cfg.guidance_scale * (ism_noise_pred_text- ipa_noise_pred_ref_text)

        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
        grad = w(self.alphas[t]) * (ipa_pred_noise - noise + self.cfg.style_guidance*(ipa_noise_pred_text - ism_noise_pred_uncond))

        guidance_eval_utils = {
                    "latents_noisy": latents_noisy,
                    "noise_pred_text": ipa_noise_pred_text
                } 

        return grad, guidance_eval_utils

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"],rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            if rgb_BCHW.shape[2]!=self.cfg.height:

                rgb_BCHW_512 = F.interpolate(
                    rgb_BCHW, (self.cfg.width, self.cfg.height), mode="bilinear", align_corners=False
                )
                # encode image into latents with vae
                latents = self.encode_images(rgb_BCHW_512)
            else:
                latents = self.encode_images(rgb_BCHW)
        return latents

    def prepare_image_cond(self, control_type, cond_rgb: Float[Tensor, "B H W C"]):
        if control_type == 'normal':
            cond_rgb = (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            detected_map = self.preprocessor_normal(cond_rgb)
            control = torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif control_type == 'canny':
            control = []
            for i in range(cond_rgb.size()[0]):
                cond_rgb_ = cond_rgb[i]
                
                cond_rgb_ = (cond_rgb_.detach().cpu().numpy() * 255).astype(np.uint8).copy()
                blurred_img = cv2.blur(cond_rgb_, ksize=(5, 5))
                detected_map = self.preprocessor_canny(blurred_img, self.cfg.canny_lower_bound,
                                                       self.cfg.canny_upper_bound,cond_rgb_.shape[0],cond_rgb_.shape[0])
                
                control_ = torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
                control.append(control_)
            control = torch.stack(control, dim=0)
            control = control.permute(0, 3, 1, 2)
        elif control_type == 'self-normal':
            control = cond_rgb.permute(0, 3, 1, 2)
        elif control_type == 'hed':
            cond_rgb = (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            detected_map = self.preprocessor_hed(cond_rgb, scribble=True)
            control = torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif control_type == 'p2p':
            control = cond_rgb.permute(0, 3, 1, 2)
        elif control_type == 'depth':
            control = cond_rgb.permute(0, 3, 1, 2)
            control = control.repeat(1, 3, 1, 1)

        if control.shape[2]!=self.cfg.height:
            return F.interpolate(
                control, (512, 512), mode="bilinear", align_corners=False
            )
        else:
            return control

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        warm_up_rate = 0,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        cond_depth = kwargs.get('cond_depth', None).permute(0, 3, 1, 2)
        depth_latent=F.interpolate(cond_depth, (64, 64), mode="bilinear", align_corners=False)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)
        #latents=rgb.permute(0, 3, 1, 2)
        if self.use_controlnet:
            image_cond = []
            condition_scales = []
            cond_rgb = kwargs.get('cond_rgb', None)
            cond_depth = kwargs.get('cond_depth', None)
            cond_normal = kwargs.get('cond_normal',None)
            for k in range(len(self.cfg.control_types)):
                control_type = self.cfg.control_types[k]
                if control_type == 'canny':
                    control_cond = self.prepare_image_cond(control_type, cond_rgb)
                elif control_type == 'depth':
                    control_cond = self.prepare_image_cond(control_type, cond_depth)
                elif control_type == 'self-normal':
                    control_cond = self.prepare_image_cond(control_type, cond_normal)
                else:
                    control_cond = self.prepare_image_cond(control_type, cond_rgb)
                image_cond.append(control_cond)

                condition_scales.append(self.cfg.condition_scales[k])
            grad, guidance_eval_utils = self.compute_grad_sds(
                prompt_utils, condition_scales, latents, depth_latent, image_cond, elevation, azimuth, camera_distances, warm_up_rate
            )
        else:
            condition_scales = [0]
            image_cond = []
            grad, guidance_eval_utils = self.compute_grad_sds(
                prompt_utils, condition_scales, latents, depth_latent, image_cond, elevation, azimuth, camera_distances, warm_up_rate
            )

        grad = torch.nan_to_num(grad * self.cfg.grad_scale)

        if self.cfg.grad_clip_val is not None:
            grad = grad.clamp(-self.cfg.grad_clip_val, self.cfg.grad_clip_val)
        if self.cfg.grad_normalize:
            grad = grad / (grad.norm(2) + 1e-8)

        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
        }
        guidance_out.update(guidance_eval_utils)

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.warmup_step = int(self.num_train_timesteps*(self.cfg.max_step_range - max_step_percent))

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.noise_scale = C(self.cfg.noise_scale, epoch, global_step)
        self.cond_scale=C(self.cfg.cond_scale, epoch, global_step)
        self.uncond_scale=C(self.cfg.uncond_scale, epoch, global_step)
        self.null_scale=C(self.cfg.null_scale, epoch, global_step)
        self.perpneg_scale=C(self.cfg.perpneg_scale, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
        
        if (
            self.use_controlnet
            and self.cfg.control_anneal_start_step is not None
            and global_step > self.cfg.control_anneal_start_step
        ):
            self.cfg.condition_scales = self.cfg.condition_scales_anneal
    
