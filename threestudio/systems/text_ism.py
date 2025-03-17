from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.ops import (
            get_mvp_matrix,
            get_projection_matrix,
            get_ray_directions,
            get_rays,
        )
import math
import random
from threestudio.utils.ops import get_activation
@threestudio.register("text_ism-system")
class Text_ISM(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        texture:bool=True
        latent_steps: int = 1000
        save_train_image: bool = True
        save_train_image_iter: int = 1
        init_step: int = 0
        init_width:int=512
        init_height:int=512
        test_background_white: Optional[bool] = False
        warmup_iter:int=1500

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        # self.init_batch = self.make_init_batch(
        self.init_batch = self.make_atlas_init_batch(
            width=self.cfg.init_width,
            height=self.cfg.init_height,
            eval_camera_distance=3.0,
            eval_fovy_deg=45.,
            eval_elevation_deg=15.0,
            n_views=4,
            eval_batch_size=4,
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_rgb=self.cfg.texture)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        # import ipdb
        # ipdb.set_trace()
        self.cfg.prompt_processor.ref_prompt = self.cfg.guidance.ref_content_prompt
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if not self.cfg.texture:
            # initialize SDF
            # FIXME: what if using other geometry types?
            self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        prompt_utils = self.prompt_processor()
        if self.true_global_step < self.cfg.init_step:
            out_0 = self(self.get_index_init_batch(0))
            out_1 = self(self.get_index_init_batch(1))
            out_2 = self(self.get_index_init_batch(2))
            out_3 = self(self.get_index_init_batch(3))
            out = dict()
            for k in out_0.keys():
                if k =='mesh' :
                    out[k] = out_0[k]
                    continue
                out[k] = torch.cat([ torch.cat([out_0[k],out_1[k]],dim=1), torch.cat([out_2[k],out_3[k]],dim=1)] , dim=2)
        else:
            out = self(batch)

        warm_up_rate = 1. - min(self.true_global_step/self.cfg.warmup_iter,1.)

        guidance_inp = out["comp_rgb"]
        # srgb=out["comp_rgb"].detach()
        # mask=out["opacity"].detach()
        # guidance_inp=torch.cat((srgb, mask), 3)

        #guidance_inp = get_activation("lin2srgb")(out["comp_rgb"])
        #guidance_inp=out["comp_rgb"].clamp(0.0, 1.0)
        batch['cond_normal']=out.get('comp_normal', None)
        batch['cond_depth']=out.get('comp_depth', None)
        #batch['condition_map'][...,1:4]=out.get('comp_normal', None)
        guidance_out = self.guidance(
            guidance_inp, prompt_utils, **batch, rgb_as_latents=False, warm_up_rate=warm_up_rate
        )
        

        loss = 0.0    
        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        def decode_latents(
            latents: Float[Tensor, "B 4 H W"],
            latent_height: int = 64,
            latent_width: int = 64,
        ) -> Float[Tensor, "B 3 512 512"]:
            input_dtype = latents.dtype
            latents = F.interpolate(
                latents, (latent_height, latent_width), mode="bilinear", align_corners=False
            )
            latents = 1 / self.guidance.vae.config.scaling_factor * latents
            image = self.guidance.vae.decode(latents.to(torch.float16)).sample
            image = (image * 0.5 + 0.5).clamp(0, 1)
            return image.to(input_dtype)
        if self.cfg.save_train_image:
            if self.true_global_step%self.cfg.save_train_image_iter == 0:
                #srgb=get_activation("lin2srgb")(out["comp_rgb"][0].detach())
                train_images_row=[
                    {
                        "type": "rgb",
                        "img": decode_latents(guidance_out['latents_noisy'][0].unsqueeze(0)).squeeze(0).permute(1,2,0),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": decode_latents(guidance_out['latents_noisy'][1].unsqueeze(0)).squeeze(0).permute(1,2,0),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": decode_latents(guidance_out['noise_pred_text']).squeeze(0).permute(1,2,0),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },

                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out["comp_depth"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                self.save_image_grid(f"train/it{self.true_global_step}.png",
                    imgs=train_images_row,
                    name="train_step",
                    step=self.true_global_step,
                )
            
        return {"loss": loss}

    def validation_step(self, batch):
        def decode_latents(
            latents: Float[Tensor, "B 4 H W"],
            latent_height: int = 64,
            latent_width: int = 64,
        ) -> Float[Tensor, "B 3 512 512"]:
            input_dtype = latents.dtype
            latents = F.interpolate(
                latents, (latent_height, latent_width), mode="bilinear", align_corners=False
            )
            latents = 1 / self.guidance.vae.config.scaling_factor * latents
            image = self.guidance.vae.decode(latents.to(torch.float16)).sample
            image = (image * 0.5 + 0.5).clamp(0, 1)
            return image.to(input_dtype)
        out = self(batch)
        #srgb=get_activation("lin2srgb")(out["comp_rgb"][0].detach())
        srgb=out["comp_rgb"].detach()
        # latents = srgb.permute(0, 3, 1, 2)
        # srgb=decode_latents(latents)
        # srgb = srgb.permute(0, 2, 3, 1)


        self.save_image_grid(
            f"validate/it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": srgb[0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [

                {
                    "type": "grayscale",
                    "img": out["comp_depth"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )



        
    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch):
        out = self(batch)
        #srgb=get_activation("lin2srgb")(out["comp_rgb"][0].detach())
        srgb=out["comp_rgb"][0].detach()
        # self.save_img(out["comp_depth"][0, :, :, 0].detach(),f"depth/{batch['index'][0]}.png")
        # self.save_img(out["comp_normal"][0].detach(),f"normal/{batch['index'][0]}.png")
        if self.true_global_step!=0:
            self.save_image_grid(
                f"it{self.true_global_step}-test/view/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": srgb,
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if self.cfg.texture
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },

                ],
                name="test_step",
                step=self.true_global_step,
            )
            
            mask=out["opacity"][0].detach()
            self.save_img(torch.cat((srgb,mask),2),f"it{self.true_global_step}-test/render/{batch['index'][0]}.png")
        else:
            self.save_img(srgb,f"it{self.true_global_step}-test/render/{batch['index'][0]}.png")


    def on_test_epoch_end(self):
        if self.true_global_step != 0:
            viewpath="it"+str(self.true_global_step)+"-test/view"
            self.save_gif(viewpath,fps=30)

#------------------------below is getting the init batch, 4 views, HARD CODING NOW (TODO)-------------------
    def make_init_batch(self,
        width=512,
        height=512,
        eval_camera_distance=1.5,
        eval_fovy_deg=70.,
        eval_elevation_deg=15.0,
        n_views=4,
        eval_batch_size=4,
        ):

        azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[: n_views]
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                                    None, :
                                    ].repeat(eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
                torch.rand(eval_batch_size)
                * (1.5 - 0.8)
                + 0.8
        )

        # sample light direction within restricted angle range (pi/3)
        local_z = F.normalize(camera_positions, dim=-1)
        local_x = F.normalize(
            torch.stack(
                [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                dim=-1,
            ),
            dim=-1,
        )
        local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
        rot = torch.stack([local_x, local_y, local_z], dim=-1)
        light_azimuth = (
                torch.rand(eval_batch_size) * math.pi - 2 * math.pi
        )  # [-pi, pi]
        light_elevation = (
                torch.rand(eval_batch_size) * math.pi / 3 + math.pi / 6
        )  # [pi/6, pi/2]
        light_positions_local = torch.stack(
            [
                light_distances
                * torch.cos(light_elevation)
                * torch.cos(light_azimuth),
                light_distances
                * torch.cos(light_elevation)
                * torch.sin(light_azimuth),
                light_distances * torch.sin(light_elevation),
            ],
            dim=-1,
        )
        light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, width / height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx, w2c= get_mvp_matrix(c2w, proj_mtx)
        
        # get directions by dividing directions_unit_focal by focal length
        directions_unit_focal = get_ray_directions(H=height, W=width, focal=1.0)
            
        focal_length: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy)

        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(eval_batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        env_id: Int[Tensor,"B"]=(torch.rand(eval_batch_size)*5).floor().long()
        return {
            "env_id": env_id.cuda(),
            "rays_o": rays_o.cuda(),
            "rays_d": rays_d.cuda(),
            'mvp_mtx': mvp_mtx.cuda(),
            'camera_positions': camera_positions.cuda(),
            'light_positions': light_positions.cuda(),
            'height': height,
            'width': width,
            "elevation": elevation_deg.cuda(),
            "azimuth": azimuth_deg.cuda(),
            "camera_distances": camera_distances.cuda(),
            "c2w": c2w.cuda(),
            "w2c": w2c.cuda(),

        }
    
    def make_atlas_init_batch(self,
        width=512,
        height=512,
        eval_camera_distance=1.5,
        eval_fovy_deg=70.,
        eval_elevation_deg=15.0,
        n_views=4,
        eval_batch_size=4,
        ):

        azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[: n_views]
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                                    None, :
                                    ].repeat(eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
                # torch.rand(eval_batch_size)
                torch.rand(1).repeat(eval_batch_size)
                * (1.5 - 0.8)
                + 0.8
        )

        # sample light direction within restricted angle range (pi/3)
        local_z = F.normalize(camera_positions, dim=-1)
        local_x = F.normalize(
            torch.stack(
                [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                dim=-1,
            ),
            dim=-1,
        )
        local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
        rot = torch.stack([local_x, local_y, local_z], dim=-1)
        light_azimuth = (
                # torch.rand(eval_batch_size) * math.pi - 2 * math.pi
                torch.rand(1).repeat(eval_batch_size)* math.pi - 2 * math.pi
        )  # [-pi, pi]
        light_elevation = (
                # torch.rand(eval_batch_size) * math.pi / 3 + math.pi / 6
                torch.rand(1).repeat(eval_batch_size) * math.pi / 3 + math.pi / 6
        )  # [pi/6, pi/2]
        light_positions_local = torch.stack(
            [
                light_distances
                * torch.cos(light_elevation)
                * torch.cos(light_azimuth),
                light_distances
                * torch.cos(light_elevation)
                * torch.sin(light_azimuth),
                light_distances * torch.sin(light_elevation),
            ],
            dim=-1,
        )
        light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, width / height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx, w2c= get_mvp_matrix(c2w, proj_mtx)
        
        # get directions by dividing directions_unit_focal by focal length
        directions_unit_focal = get_ray_directions(H=height, W=width, focal=1.0)
            
        focal_length: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy)

        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(eval_batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        # env_id: Int[Tensor,"B"]=(torch.rand(eval_batch_size)*5).floor().long()
        env_id: Int[Tensor,"B"]=(torch.rand(1).repeat(eval_batch_size)*5).floor().long()
        return {
            "env_id": env_id.cuda(),
            "rays_o": rays_o.cuda(),
            "rays_d": rays_d.cuda(),
            'mvp_mtx': mvp_mtx.cuda(),
            'camera_positions': camera_positions.cuda(),
            'light_positions': light_positions.cuda(),
            'height': height,
            'width': width,
            "elevation": elevation_deg.cuda(),
            "azimuth": azimuth_deg.cuda(),
            "camera_distances": camera_distances.cuda(),
            "c2w": c2w.cuda(),
            "w2c": w2c.cuda(),

        }


    def get_index_init_batch(self,index):
        i = index
        return {
                'rays_o':self.init_batch['rays_o'][i:i + 1],
                'rays_d':self.init_batch['rays_d'][i:i + 1],
                'mvp_mtx': self.init_batch['mvp_mtx'][i:i + 1],
                'camera_positions': self.init_batch['camera_positions'][i:i + 1],
                'light_positions': self.init_batch['light_positions'][i:i + 1],
                'height': self.init_batch['height'],
                'width': self.init_batch['width'],
                "elevation": self.init_batch['elevation'][i:i + 1],
                "azimuth": self.init_batch['azimuth'][i:i + 1],
                "camera_distances": self.init_batch['camera_distances'][i:i + 1],
                "c2w": self.init_batch['c2w'][i:i + 1],
                "w2c": self.init_batch['w2c'][i:i + 1],
                }
    def get_random_init_batch(self):
        views = len(self.init_batch['mvp_mtx'])
        i = random.randint(0, views-1)

        return {
                'rays_o':self.init_batch['rays_o'][i:i + 1],
                'rays_d':self.init_batch['rays_d'][i:i + 1],
                'mvp_mtx': self.init_batch['mvp_mtx'][i:i + 1],
                'camera_positions': self.init_batch['camera_positions'][i:i + 1],
                'light_positions': self.init_batch['light_positions'][i:i + 1],
                'height': self.init_batch['height'],
                'width': self.init_batch['width'],
                "elevation": self.init_batch['elevation'][i:i + 1],
                "azimuth": self.init_batch['azimuth'][i:i + 1],
                "camera_distances": self.init_batch['camera_distances'][i:i + 1],
                "c2w": self.init_batch['c2w'][i:i + 1],
                "w2c": self.init_batch['w2c'][i:i + 1],
                }