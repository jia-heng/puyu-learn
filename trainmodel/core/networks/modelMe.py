import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convunet import ConvUNet, ConvUNet_L, ConvUNet_S, ConvUNet_T, ConvUNet_F
from .partitioning_pyramid import PartitioningPyramid, PartitioningPyramid_Large, PartitioningPyramid_Small, PartitioningPyramid_Tiny
from ..loss.loss import Features, SMAPE, BaseLoss
from ..util import normalize_radiance, clip_logp1, dist_cat, backproject_pixel_centers
import os
import pyexr
import numpy as np
from threading import Thread

class BaseModel(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1)
        )

        self.filter = PartitioningPyramid()
        self.weight_predictor = ConvUNet(
            70,
            self.filter.inputs
        )

        # self.features = Features(transfer='pu') # Broken with pytorch
        self.features = Features(transfer='log')

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.94)
        return [opt], [sched]

    def tensor_like(self, like, data):
        return torch.tensor(data, dtype=like.dtype, device=like.device)

    def create_meshgrid(self, motion):
        batchsize, channel, height, width = motion.shape
        dtype = motion.dtype
        device = motion.device
        y = torch.arange(0, height, dtype=dtype, device=device)
        x = torch.arange(0, width, dtype=dtype, device=device)
        Y, X = torch.meshgrid(y, x)
        meshgrid = torch.stack((X, Y), dim=-1)

        grid = meshgrid - motion.permute(0, 2, 3, 1) + 0.5
        grid = grid / self.tensor_like(grid, [width, height]) * 2 - 1

        return grid

    def step(self, x, temporal):
        grid = self.create_meshgrid(x['motion'])
        reprojected = F.grid_sample(
            temporal, # permute(0, 3, 1, 2)
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        prev_color = reprojected[:, :3]
        prev_output = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]
        # self.saveWarp(x['frame_index'].item(), prev_color, "warps")
        # Sample encoder

        color = x['color']
        batch_size = color.shape[0]
        encoder_input = torch.concat((
            x['depth'],
            x['normal'],
            x['albedo'],
            clip_logp1(normalize_radiance(color))
        ), 1)
        feature = self.encoder(encoder_input)
        # feature = self.encoder(torch.permute(encoder_input, (0, 4, 1, 2, 3)).flatten(0, 1))
        # feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1)
        # Denoiser

        weight_predictor_input = torch.concat((
            clip_logp1(normalize_radiance(torch.concat((
                prev_color,
                color
            ), 1))),
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        # weights = [weight.to(torch.float32) for weight in weights]
        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color, prev_output)
        return output, torch.concat((
            color,
            output,
            feature
        ), 1), grid

    def one_step_loss(self, ref, pred):
        ref, mean = normalize_radiance(ref, True)
        pred = pred * mean

        spatial = SMAPE(ref, pred) * 0.8 * 10 + \
                  self.features.spatial_loss(ref, pred) * 0.5 * 0.2

        return spatial, pred

    def two_step_loss(self, refs, preds, grid):
        refs, mean = normalize_radiance(refs, True)
        preds = preds * mean

        prev_ref = F.grid_sample(
            refs[:, 0],
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        prev_pred = F.grid_sample(
            preds[:, 0],
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        spatial = SMAPE(refs.flatten(0, 1), preds.flatten(0, 1)) * 2.0 + \
                  self.features.spatial_loss(refs.flatten(0, 1), preds.flatten(0, 1)) * 0.025

        diff_ref = refs[:, 1] - prev_ref
        diff_pred = preds[:, 1] - prev_pred

        temporal = SMAPE(diff_ref, diff_pred) * 0.2 + \
                   self.features.temporal_loss(refs[:, 1], preds[:, 1], prev_ref, prev_pred) * 0.025

        return spatial + temporal, preds[:, 1]

    def temporal_init(self, x):
        shape = list(x['color'].shape)
        shape[1] = 38
        return torch.zeros(shape, dtype=x['color'].dtype, device=x['color'].device)

    def bptt_step(self, x):
        if x['frame_index'] == 0:
            # with torch.no_grad():
            temporal = self.temporal_init(x)
            y, _, _ = self.step(x, temporal)
            loss, y = self.one_step_loss(x['reference'], y)
            # loss = None
        else:
            y_1, temporal, _ = self.step(self.prev_x, self.temporal)
            y_2, _, grid = self.step(x, temporal)
            loss, y = self.two_step_loss(
                torch.stack((self.prev_x['reference'], x['reference']), 1),
                torch.stack((y_1, y_2), 1),
                grid
            )
            # if loss > 0.5:
            #     print(x['file'] + ' loss:', loss)
        self.prev_x = x
        self.temporal = temporal.detach()
        return loss, y

    def training_step(self, x, batch_idx):
        loss, y = self.bptt_step(x)
        if loss:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, x, batch_idx):
        loss, y = self.bptt_step(x)
        # path = r'.\test\temp'
        if x['frame_index'] == 63:
            # self.save_exr(self.current_epoch, y.detach(), path, 'validation')
            y = normalize_radiance(y)
            y = torch.pow(y / (y + 1), 1 / 2.2)
            y = dist_cat(y).cpu().numpy()

            # Writing images can block the rank 0 process for a couple seconds
            # which often breaks distributed training so we start a separate thread
            Thread(target=self.save_images, args=(y, batch_idx, self.trainer.current_epoch)).start()
        if loss:
            self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, x):
        y, temporal, _ = self.step(x, self.temporal)
        self.temporal = temporal.detach()
        return y

    def save_exr(self, idx, image, path, imgType):
        # image = np.transpose(image.cpu().numpy()[0], (1,2,0))
        # image = (ACES(image)*255).astype(np.uint8)
        # self.save_pool.apply_async(iio.imwrite, [file, image])
        imgType = str(imgType)
        output_path = os.path.join(path, 'exr')
        os.makedirs(output_path, exist_ok=True)
        filename = '{imgType}_{idx:04d}.exr'.format(imgType=imgType, idx=idx)
        file_path = os.path.join(output_path, filename)
        image_array = np.transpose(image.cpu().numpy()[0], (1, 2, 0))
        pyexr.write(file_path, image_array)

    def save_images(self, images, batch_idx, epoch):
        for i, image in enumerate(images):
            self.logger.experiment.add_image(f'denoised/{batch_idx}-{i}', image, epoch)

class model_kernel_nppd(BaseModel):
    def step(self, x, temporal):
        grid = backproject_pixel_centers(
            torch.mean(x['motion'], -1),
            x['crop_offset'],
            x['prev_camera']['crop_offset'],
            as_grid=True
        )

        reprojected = F.grid_sample(
            temporal,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        prev_color = reprojected[:, :3]
        prev_output = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]

        # Sample encoder

        batch_size = x['color'].shape[0]

        encoder_input = torch.concat((
            x['depth'],
            x['normal'],
            x['diffuse'],
            clip_logp1(normalize_radiance(x['color']))
        ), 1)

        feature = self.encoder(
            torch.permute(encoder_input, (0, 4, 1, 2, 3)).flatten(0, 1)
        )
        # feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1).to(torch.float32)
        feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1)
        color = torch.mean(x['color'], -1)

        # Denoiser

        weight_predictor_input = torch.concat((
            clip_logp1(normalize_radiance(torch.concat((
                prev_color,
                color
            ), 1))),
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        # weights = [weight.to(torch.float32) for weight in weights]

        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color, prev_output)

        return output, torch.concat((
            color,
            output,
            feature
        ), 1), grid

    def temporal_init(self, x):
        shape = list(x['reference'].shape)
        shape[1] = 38
        return torch.zeros(shape, dtype=x['reference'].dtype, device=x['reference'].device)

class model_kernel_nppd_mean(model_kernel_nppd):
    def step(self, x, temporal):
        grid = backproject_pixel_centers(
            x['motion'],
            x['crop_offset'],
            x['prev_camera']['crop_offset'],
            as_grid=True
        )

        reprojected = F.grid_sample(
            temporal,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        prev_color = reprojected[:, :3]
        prev_output = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]

        color = x['color']
        batch_size = color.shape[0]
        encoder_input = torch.concat((
            x['depth'],
            x['normal'],
            x['diffuse'],
            clip_logp1(normalize_radiance(x['color']))
        ), 1)

        feature = self.encoder(encoder_input)
        #     torch.permute(encoder_input, (0, 4, 1, 2, 3)).flatten(0,1)
        # )
        #
        # feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1)
        # color = torch.mean(x['color'], -1)

        weight_predictor_input = torch.concat((
            clip_logp1(normalize_radiance(torch.concat((
                prev_color,
                color
            ), 1))),
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        # weights = [weight.to(torch.float32) for weight in weights]
        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color, prev_output)
        return output, torch.concat((
            color,
            output,
            feature
        ), 1), grid

class model_kernel_L(BaseModel):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1)
        )

        self.filter = PartitioningPyramid_Large()
        self.weight_predictor = ConvUNet_L(
            70,
            self.filter.inputs
        )
        self.features = Features(transfer='log')

    def step(self, x, temporal):
        grid = self.create_meshgrid(x['motion'])
        reprojected = F.grid_sample(
            temporal, # permute(0, 3, 1, 2)
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        prev_color = reprojected[:, :3]
        prev_output = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]

        color = x['color']
        batch_size = color.shape[0]
        encoder_input = torch.concat((
            x['depth'],
            x['normal'],
            x['albedo'],
            clip_logp1(normalize_radiance(color))
        ), 1)
        feature = self.encoder(encoder_input)
        # feature = self.encoder(torch.permute(encoder_input, (0, 4, 1, 2, 3)).flatten(0, 1))
        # feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1)
        # Denoiser

        weight_predictor_input = torch.concat((
            clip_logp1(normalize_radiance(torch.concat((
                prev_color,
                color
            ), 1))),
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        # weights = [weight.to(torch.float32) for weight in weights]
        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color, prev_output)
        return output, torch.concat((
            color,
            output,
            feature
        ), 1), grid

class model_kernel_S(BaseModel):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1)
        )

        self.filter = PartitioningPyramid_Small()
        self.weight_predictor = ConvUNet_S(
            70,
            self.filter.inputs
        )
        self.features = Features(transfer='log')

    def step(self, x, temporal):
        grid = self.create_meshgrid(x['motion'])
        reprojected = F.grid_sample(
            temporal, # permute(0, 3, 1, 2)
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        prev_color = reprojected[:, :3]
        prev_output = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]

        color = x['color']
        batch_size = color.shape[0]
        encoder_input = torch.concat((
            x['depth'],
            x['normal'],
            x['albedo'],
            clip_logp1(normalize_radiance(color))
        ), 1)
        feature = self.encoder(encoder_input)
        # feature = self.encoder(torch.permute(encoder_input, (0, 4, 1, 2, 3)).flatten(0, 1))
        # feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1)
        # Denoiser

        weight_predictor_input = torch.concat((
            clip_logp1(normalize_radiance(torch.concat((
                prev_color,
                color
            ), 1))),
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        # weights = [weight.to(torch.float32) for weight in weights]
        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color, prev_output)
        return output, torch.concat((
            color,
            output,
            feature
        ), 1), grid

class model_kernel_S_nppd(model_kernel_S):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1)
        )

        self.filter = PartitioningPyramid_Small()
        self.weight_predictor = ConvUNet_S(
            70,
            self.filter.inputs
        )
        self.features = Features(transfer='log')

    def step(self, x, temporal):
        grid = backproject_pixel_centers(
            x['motion'],
            x['crop_offset'],
            x['prev_camera']['crop_offset'],
            as_grid=True
        )

        reprojected = F.grid_sample(
            temporal,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        prev_color = reprojected[:, :3]
        prev_output = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]

        color = x['color']
        batch_size = color.shape[0]
        encoder_input = torch.concat((
            x['depth'],
            x['normal'],
            x['diffuse'],
            clip_logp1(normalize_radiance(x['color']))
        ), 1)

        feature = self.encoder(encoder_input)
        #     torch.permute(encoder_input, (0, 4, 1, 2, 3)).flatten(0,1)
        # )
        #
        # feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1)
        # color = torch.mean(x['color'], -1)

        weight_predictor_input = torch.concat((
            clip_logp1(normalize_radiance(torch.concat((
                prev_color,
                color
            ), 1))),
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        # weights = [weight.to(torch.float32) for weight in weights]
        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color, prev_output)
        return output, torch.concat((
            color,
            output,
            feature
        ), 1), grid

    def temporal_init(self, x):
        shape = list(x['reference'].shape)
        shape[1] = 38
        return torch.zeros(shape, dtype=x['reference'].dtype, device=x['reference'].device)

class model_kernel_T(model_kernel_S):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1)
        )

        self.filter = PartitioningPyramid_Small()
        self.weight_predictor = ConvUNet_T(
            70,
            self.filter.inputs
        )
        self.features = Features(transfer='log')

class model_kernel_T_nppd(model_kernel_S_nppd):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1)
        )

        self.filter = PartitioningPyramid_Small()
        self.weight_predictor = ConvUNet_T(
            70,
            self.filter.inputs
        )
        self.features = Features(transfer='log')

class model_kernel_T_B(model_kernel_T):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1)
        )

        self.filter = PartitioningPyramid_Tiny()
        self.weight_predictor = ConvUNet_T(
            64,
            self.filter.inputs
        )
        self.features = Features(transfer='log')
    def step(self, x, temporal):
        grid = self.create_meshgrid(x['motion'])
        reprojected = F.grid_sample(
            temporal, # permute(0, 3, 1, 2)
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        prev_output = reprojected[:, :3]
        prev_color = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]

        color = x['color']
        encoder_input = torch.concat((
            clip_logp1(normalize_radiance(color)),
            x['depth'],
            x['normal'],
            x['albedo']
        ), 1)

        feature = self.encoder(encoder_input)
        weight_predictor_input = torch.concat((
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color, prev_output)
        return output, torch.concat((
            output,
            color,
            feature
        ), 1), grid

class model_kernel_T_B_nppd(model_kernel_T_B):
    def step(self, x, temporal):
        grid = backproject_pixel_centers(
            x['motion'],
            x['crop_offset'],
            x['prev_camera']['crop_offset'],
            as_grid=True
        )

        reprojected = F.grid_sample(
            temporal,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        prev_output = reprojected[:, :3]
        prev_color = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]

        color = x['color']
        encoder_input = torch.concat((
            clip_logp1(normalize_radiance(x['color'])),
            x['depth'],
            x['normal'],
            x['diffuse']
        ), 1)

        feature = self.encoder(encoder_input)

        weight_predictor_input = torch.concat((
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color, prev_output)
        return output, torch.concat((
            output,
            color,
            feature
        ), 1), grid

class model_kernel_F_nppd(model_kernel_T_B_nppd):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 13, 1)
        )

        self.filter = PartitioningPyramid_Tiny()
        self.weight_predictor = ConvUNet_F(
            32,
            self.filter.inputs
        )

        self.features = Features(transfer='log')

    def step(self, x, temporal):
        grid = backproject_pixel_centers(
            x['motion'],
            x['crop_offset'],
            x['prev_camera']['crop_offset'],
            as_grid=True
        )

        reprojected = F.grid_sample(
            temporal,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        prev_color = reprojected[:, :3]
        prev_output = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]

        color = x['color']
        norm_color, mean = normalize_radiance(color, True)
        encoder_input = torch.concat((
            clip_logp1(norm_color),
            x['depth'],
            x['normal'],
            x['diffuse']
        ), 1)

        feature = self.encoder(encoder_input)

        weight_predictor_input = torch.concat((
            clip_logp1(prev_color*mean),
            clip_logp1(norm_color),
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color, prev_output)
        return output, torch.concat((
            color,
            output,
            feature
        ), 1), grid

    def temporal_init(self, x):
        shape = list(x['color'].shape)
        shape[1] = 19
        return torch.zeros(shape, dtype=x['color'].dtype, device=x['color'].device)

class model_kernel_F(model_kernel_F_nppd):
    def step(self, x, temporal):
        grid = self.create_meshgrid(x['motion'])
        # reprojected = bilinear_grid_sample(temporal, grid, align_corners=False)
        reprojected = F.grid_sample(
            temporal, # permute(0, 3, 1, 2)
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        prev_output = reprojected[:, :3]
        prev_color = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]

        color = x['color']

        encoder_input = torch.concat((
            clip_logp1(normalize_radiance(color)),
            x['depth'],
            x['normal'],
            x['albedo']
        ), 1)

        feature = self.encoder(encoder_input)
        weight_predictor_input = torch.concat((
            clip_logp1(normalize_radiance(torch.concat((
                prev_color,
                color
            ), 1))),
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color_mix = t_lambda * prev_color + (1 - t_lambda) * color
        feature_mix = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color_mix, prev_output)
        return output, torch.concat((
            output,
            color_mix,
            feature_mix
        ), 1), grid

class model_kernel_F_T_nppd(model_kernel_F_nppd):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 24, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(24, 24, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(24, 5, 1)
        )

        self.filter = PartitioningPyramid_Tiny(3)
        self.weight_predictor = ConvUNet_F(
            16,
            self.filter.inputs
        )

    def temporal_init(self, x):
        shape = list(x['color'].shape)
        shape[1] = 11
        return torch.zeros(shape, dtype=x['color'].dtype, device=x['color'].device)

class model_kernel_W_nppd(model_kernel_F_nppd):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 24, 1)
        )

        self.filter = PartitioningPyramid_Tiny()
        self.weight_predictor = ConvUNet_F(
            54,
            self.filter.inputs
        )
        self.features = Features(transfer='log')
