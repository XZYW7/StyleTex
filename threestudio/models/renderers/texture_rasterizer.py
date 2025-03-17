from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.utils.ops import get_activation
def xfm_vectors(vectors, matrix):
    '''Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)

    Returns:
        Transformed vectors in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    

    out = torch.matmul(torch.nn.functional.pad(vectors, pad=(0,1), mode='constant', value=0.0), torch.transpose(matrix, 1, 2))[..., 0:3].contiguous()
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_vectors contains inf or NaN"
    return out
@threestudio.register("texture_rasterizer")
class textureRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        #batch_idx,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        c2w: Float[Tensor, "B 4 4"],
        w2c: Float[Tensor, "B 4 4"],
        height: int,
        width: int,
        render_normal: bool = True,
        render_rgb: bool = True,
        render_xyz : bool = False,
        render_depth: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        if render_depth:
            render_xyz = True
        if render_xyz:
            render_normal = True

        batch_size = mvp_mtx.shape[0]
        mesh = self.geometry.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        #out = {"opacity": mask_aa, "mesh": mesh}
        out = {"opacity": mask_aa, "mesh": mesh, "face_id": rast[..., 3:]}

        if render_normal:
            selector = mask[..., 0].reshape(batch_size,width*height)
            gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal=gb_normal.reshape(batch_size,width*height,3)
            normal_controlnet=self.compute_controlnet_normals(gb_normal[selector],w2c,batch_size)
            background=torch.tensor([0.5,0.5,1.0]).reshape(1,1,1,3).repeat(batch_size,width,height,1).to(self.device)
            #background=torch.ones((batch_size,width,height,3)).to(self.device)
            gb_normal_aa=torch.ones_like(gb_normal).to(self.device)
            gb_normal_aa[selector]=normal_controlnet
            gb_normal_aa=gb_normal_aa.reshape(batch_size,height,width,3)
            gb_normal_aa = torch.lerp(background, gb_normal_aa, mask.float())
            gb_normal_aa = self.ctx.antialias(gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx)
            out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        if render_rgb:
            selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]
            geo_out = self.geometry(positions, output_normal=False)

            gb_normal=gb_normal.reshape(batch_size,height,width,3)
            extra_geo_info = {}
            if self.material.requires_normal:
                
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]


            outputs = self.material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                light_positions=gb_light_positions[selector],
                **extra_geo_info,
                **geo_out
            )
            rgb_fg=outputs
            gb_rgb_fg = torch.ones(batch_size, height, width, 3).to(rgb_fg)
            gb_rgb_fg[selector] = rgb_fg

            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg})

        if render_xyz:
            # xyz map
            world_coordinates, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            out.update({"comp_xyz": world_coordinates})
            # true normal map
            gb_normal_normalize = torch.lerp(
            torch.zeros_like(gb_normal), gb_normal, mask.float()
            )
            out.update({"comp_normal_normalize": gb_normal_normalize})  # in [-1, 1]

        if render_depth:
            # calculate w2c from c2w: R' = Rt, t' = -Rt * t
            # mathematically equivalent to (c2w)^-1

            w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
            w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
            w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
            w2c[:, 3, 3] = 1.0
            # render depth
            world_coordinates_homogeneous = torch.cat([world_coordinates, torch.ones_like(world_coordinates[..., :1])], dim=-1) # shape: [batch_size, height, width, 4]
            camera_coordinates_homogeneous = torch.einsum('bijk,bkl->bijl', world_coordinates_homogeneous, w2c.transpose(-2, -1)) # shape: [batch_size, height, width, 4]
            camera_coordinates = camera_coordinates_homogeneous[..., :3] # shape: [batch_size, height, width, 3]
            depth = camera_coordinates[..., 2] # shape: [batch_size, height, width]

            mask_depth = mask.squeeze(2).squeeze(3)
            foreground_depth = depth[mask_depth]

            if foreground_depth.numel() > 0:
                min_depth = torch.min(foreground_depth)
                max_depth = torch.max(foreground_depth)
                normalized_depth = (depth - min_depth) / (max_depth - min_depth+1e-6)
            else:
                normalized_depth = (depth)*0

            background_value = 0
            depth_blended = normalized_depth * mask_depth.float() + background_value * (1 - mask_depth.float())

            out.update({"comp_depth": depth_blended.unsqueeze(3)})

        return out

    def compute_controlnet_normals(self,normals,mv,batch_size):
        normal_view  = xfm_vectors(normals.view(batch_size, normals.shape[0], normals.shape[1]), mv).view(*normals.shape)
        normal_view = F.normalize(normal_view)
        normal_controlnet=0.5*(normal_view+1)
        normal_controlnet[..., 0]=1.0-normal_controlnet[..., 0] # Flip the sign on the x-axis to match bae system
        return normal_controlnet