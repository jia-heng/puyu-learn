import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convunet import ConvUNet
from .partitioning_pyramid import PartitioningPyramid, PartitioningPyramid_Large, PartitioningPyramid_Small
from .modelMe import BaseModel, model_kernel_S_nppd
from ..loss.loss import Features, SMAPE, BaseLoss
from ..util import normalize_radiance, clip_logp1, dist_cat, backproject_pixel_centers
import os
import pyexr
import numpy as np
from threading import Thread

def reversedl(it):
    return list(reversed(list(it)))

class UNetSplat(nn.Module):
    def __init__(self, in_channels, K=5):
        super().__init__()
        self.K = K
        # 9splat 1part 1α 1pre_α 9pre_splat
        out_chans = [9 + 1 + 1 + 1 + 9] + [15 for i in range(K - 1)] # 9splat 1part 1α 4upsample
        self.partial = 9
        self.t_lambda_index = 10
        self.pre_lambda_index = 11
        self.ps = nn.PixelShuffle(2)
        self.pool = F.max_pool2d
        dims_and_depths_ds = [
            (32, 32),
            (64,),
            (96,),
            (144,),
            (248,),
            (384, 384)
        ]
        dims_and_depths_us = [
            (32, 32),
            (64, 48),
            (96, 96),
            (144, 128),
            (248, 224)
        ]
        self.K = len(out_chans)
        self.ds_path = nn.ModuleList()
        prev_dim = in_channels
        for dims in dims_and_depths_ds:
            layers = []
            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))

        self.us_path = nn.ModuleList()
        for dims in reversedl(dims_and_depths_us):
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[0], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]
            layers.append(nn.Conv2d(prev_dim, prev_dim, 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(out_chans, dims_and_depths_us):
            self.out_path.append(nn.Conv2d(dims[-1], out_chan, 1))

        self.sr_path = nn.ModuleList()

    # def forward(self, x, prev_color, color, previous, prev_feature, feature):
    #     skips = []
    #     for i in range(self.K):
    #         x = self.ds_path[i](x)
    #         skips.append(x)
    #         x = self.pool(x, 2)
    #     x = self.ds_path[self.K](x)
    #     skips.append(x)
    #
    #     x = skips[-1]
    #     i = 0
    #     j = self.K - i - 1
    #     x = torch.cat((F.interpolate(x, scale_factor=2), skips[j]), 1)
    #     # x = torch.cat((self.ps(self.ps_path[i](x)), skips[j]), 1)
    #     x = self.us_path[i](x)
    #     weight = self.out_path[j](x)
    #     t_lambda = weight[:, self.t_lambda_index, None]
    #     rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
    #     partition = weight[:, self.partial, None] * rendering
    #     denoising = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3)
    #     denoising_us = upscale_Small(denoising, F.softmax(weight[:, 11:15], 1) * 4)
    #
    #     i = 1
    #     j = self.K - i - 1
    #     x = torch.cat((F.interpolate(x, scale_factor=2), skips[j]), 1)
    #     # x = torch.cat((self.ps(self.ps_path[i](x)), skips[j]), 1)
    #     x = self.us_path[i](x)
    #     weight = self.out_path[j](x)
    #     t_lambda = weight[:, self.t_lambda_index, None]
    #     rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
    #     partition = weight[:, self.partial, None] * rendering
    #     denoising = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3) + denoising_us
    #     denoising_us = upscale_Small(denoising, F.softmax(weight[:, 11:15], 1) * 4)
    #
    #     i = 2
    #     j = self.K - i - 1
    #     x = torch.cat((F.interpolate(x, scale_factor=2), skips[j]), 1)
    #     # x = torch.cat((self.ps(self.ps_path[i](x)), skips[j]), 1)
    #     x = self.us_path[i](x)
    #     weight = self.out_path[j](x)
    #     t_lambda = weight[:, self.t_lambda_index, None]
    #     rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
    #     partition = weight[:, self.partial, None] * rendering
    #     denoising = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3) + denoising_us
    #     denoising_us = upscale_Small(denoising, F.softmax(weight[:, 11:15], 1) * 4)
    #
    #     i = 3
    #     j = self.K - i - 1
    #     x = torch.cat((F.interpolate(x, scale_factor=2), skips[j]), 1)
    #     # x = torch.cat((self.ps(self.ps_path[i](x)), skips[j]), 1)
    #     x = self.us_path[i](x)
    #     weight = self.out_path[j](x)
    #     t_lambda = weight[:, self.t_lambda_index, None]
    #     rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
    #     partition = weight[:, self.partial, None] * rendering
    #     denoising = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3) + denoising_us
    #     denoising_us = upscale_Small(denoising, F.softmax(weight[:, 11:15], 1) * 4)
    #
    #     i = 4
    #     j = self.K - i - 1
    #     x = torch.cat((F.interpolate(x, scale_factor=2), skips[j]), 1)
    #     # x = torch.cat((self.ps(self.ps_path[i](x)), skips[j]), 1)
    #     x = self.us_path[i](x)
    #     weight = self.out_path[j](x)
    #     t_lambda = weight[:, self.t_lambda_index, None]
    #     rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
    #     partition = weight[:, self.partial, None] * rendering
    #     denoised = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3) + denoising_us
    #
    #     previous = splat_unfold(previous, F.softmax(weight[:, -9:], 1), 3)
    #     t_mu = torch.sigmoid(weight[:, self.pre_lambda_index, None])
    #     predict = t_mu * previous + (1 - t_mu) * denoised
    #     color_ac = rendering
    #     feature_ac = t_lambda * prev_feature + (1 - t_lambda) * feature
    #
    #     # SR
    #     i = 5
    #     x = torch.cat((x, skips[-1]), 1)
    #     x = self.us_path[i](x)
    #     x = torch.cat((x, predict), 1)
    #     output = self.sr_path(F.interpolate(x, scale_factor=2))
    #
    #     return predict, color_ac, feature_ac

class UNet_SF(nn.Module):
    def __init__(self, in_channels=23, out_channels=16):
        super(UNet_SF, self).__init__()
        out_chans = 2  # 9splat 1part 1α 4upsample
        self.pool = F.max_pool2d
        self.K = 4
        dims_and_depths_ds = [
            (32, 32),
            (48,),
            (64,),
            (80,),
            (96, 96)
        ]
        dims_and_depths_us = [
            (in_channels, 64, 32, out_channels),
            (32, 64),
            (48, 96),
            (64, 112)
        ]

        self.ds_path = nn.ModuleList()
        prev_dim = in_channels
        for dims in dims_and_depths_ds:
            layers = []
            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding=1))
                layers.append(nn.ReLU(True))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))

        self.us_path = nn.ModuleList()
        for dims in reversedl(dims_and_depths_us[1:]):
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[0], dims[-1], 3, padding=1))
            layers.append(nn.ReLU(True))
            prev_dim = dims[-1]
            layers.append(nn.Conv2d(prev_dim, prev_dim, 3, padding=1))
            layers.append(nn.ReLU(True))

            self.us_path.append(nn.Sequential(*layers))

        dims = dims_and_depths_us[0]
        layers = []
        layers.append(nn.Conv2d(prev_dim+dims[0], dims[1], 3, padding=1))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(dims[1], dims[2], 3, padding=1))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(dims[2], dims[3], 3, padding=1))

        self.us_path.append(nn.Sequential(*layers))

    def forward(self, x):
        skips = []
        skips.append(x)
        for i in range(self.K):
            x = self.ds_path[i](x)
            x = self.pool(x, 2)
            skips.append(x)
        x = self.ds_path[self.K](x)
        skips.append(x)

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[2:]):
            x = torch.cat((F.interpolate(x, scale_factor=2, mode='nearest'), skip), 1)
            x = stage(x)

        return x

class model_SF(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.filter = UNet_SF(23, 13)
        self.features = Features(transfer='log')

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.96)
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
        ''' temporal: precolor preoutput prefeature 13 '''
        grid = self.create_meshgrid(x['motion'])
        reprojected = F.grid_sample(
            temporal,  # permute(0, 3, 1, 2)
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
        feature = torch.concat((
            x['depth'],
            x['normal'],
            x['albedo'],
            clip_logp1(normalize_radiance(color))
        ), 1)

        filter_input = torch.concat((
            prev_output,
            clip_logp1(normalize_radiance(prev_color)),
            prev_feature,
            clip_logp1(normalize_radiance(color)),
            feature
        ), 1)

        output = self.filter(filter_input)
        predict = output[:, :3]

        return predict, output, grid

    def temporal_init(self, x):
        return torch.concat((
            x['color'],
            x['color'],
            x['depth'],
            x['normal'],
            x['albedo']
        ), 1)

    def one_step_loss(self, ref, pred):
        ref, mean = normalize_radiance(ref, True)
        pred = pred / mean

        spatial = SMAPE(ref, pred) * 0.8 * 10 + \
                  self.features.spatial_loss(ref, pred) * 0.5 * 0.2

        return spatial, pred

    def two_step_loss(self, refs, preds, grid):
        refs, mean = normalize_radiance(refs, True)
        preds = preds / mean

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

class model_SF_nppd(model_SF):
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.94)
        return [opt], [sched]

    def temporal_init(self, x):
        return torch.concat((
            x['color'],
            x['color'],
            x['depth'],
            x['normal'],
            x['diffuse']
        ), 1)

    def step(self, x, temporal):
        ''' temporal: precolor preoutput prefeature 13 '''
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
        feature = torch.concat((
            x['depth'],
            x['normal'],
            x['diffuse']
        ), 1)

        filter_input = torch.concat((
            prev_output,
            clip_logp1(normalize_radiance(prev_color)),
            prev_feature,
            clip_logp1(normalize_radiance(color)),
            feature
        ), 1)

        output = self.filter(filter_input)
        predict = output[:, :3]

        return predict, output, grid


class model_kernel_init(L.LightningModule):
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
        # self.features = Features(transfer='log')

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

    def forward(self, color, depth, normal, albedo, motion, temporal):
        grid = self.create_meshgrid(motion)
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

        encoder_input = torch.concat((
            depth,
            normal,
            albedo,
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
        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color_mix = t_lambda * prev_color + (1 - t_lambda) * color
        feature_mix = t_lambda * prev_feature + (1 - t_lambda) * feature
        predict = self.filter(weights, color_mix, prev_output)
        output2 = torch.concat((color_mix, predict, feature_mix), 1)
        return predict, output2


class model_kernel_SF(model_kernel_init):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1)
        )

        self.filter = UNetSplat(70)
        # self.features = Features(transfer='log')
    def forward(self, color, depth, normal, albedo, motion, temporal):
        grid = self.create_meshgrid(motion)
        # reprojected = bilinear_grid_sample(temporal, grid, align_corners=False)
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

        encoder_input = torch.concat((
            depth,
            normal,
            albedo,
            color
        ), 1)
        feature = self.encoder(encoder_input)

        weight_predictor_input = torch.concat((
            prev_color,
            color,
            prev_feature,
            feature
        ), 1)

        predict, color_ac, feature_ac = self.filter(weight_predictor_input, prev_color, color, prev_output, prev_feature, feature)
        output2 = torch.concat((color_ac, predict, feature_ac), 1)

        return predict, output2
