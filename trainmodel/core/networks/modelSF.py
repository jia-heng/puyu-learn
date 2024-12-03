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

def Conv(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, 3, padding=1)

# ReLU function
def relu(x):
  return F.relu(x, inplace=True)

# 2x2 max pool function
def pool(x):
  return F.max_pool2d(x, 2, 2)

# 2x2 nearest-neighbor upsample function
def upsample(x):
  return F.interpolate(x, scale_factor=2, mode='nearest')

# Channel concatenation function
def concat(a, b):
  return torch.cat((a, b), 1)

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

        # ic = in_channels
        # ec1  = 32
        # ec2  = 48
        # ec3  = 64
        # ec4  = 80
        # ec5  = 96
        #
        # dc4  = 112
        # dc3  = 96
        # dc2a = 64
        # dc2b = 64
        # dc1a = 64
        # dc1b = 32
        #
        # oc = out_channels
        #
        # # Convolutions
        # self.enc_conv0  = Conv(ic,      ec1) # ic 32
        # self.enc_conv1  = Conv(ec1,     ec1) # 32 32
        # self.enc_conv2  = Conv(ec1,     ec2) # 32 48
        # self.enc_conv3  = Conv(ec2,     ec3) # 48 64
        # self.enc_conv4  = Conv(ec3,     ec4) # 64 80
        # self.enc_conv5a = Conv(ec4,     ec5) # 80 96
        # self.enc_conv5b = Conv(ec5,     ec5) # 96 96
        #
        # self.dec_conv4a = Conv(ec5+ec3, dc4) # 96+64 112
        # self.dec_conv4b = Conv(dc4,     dc4) # 112 112
        # self.dec_conv3a = Conv(dc4+ec2, dc3) # 112+48 96
        # self.dec_conv3b = Conv(dc3,     dc3) # 96 96
        # self.dec_conv2a = Conv(dc3+ec1, dc2a) # 96+32 64
        # self.dec_conv2b = Conv(dc2a,    dc2b) # 64 64
        # self.dec_conv1a = Conv(dc2b+ic, dc1a) # 64+ic 64
        # self.dec_conv1b = Conv(dc1a,    dc1b) # 64 32
        # self.dec_conv0  = Conv(dc1b,    oc) # 32 oc
        #
        # # Images must be padded to multiples of the alignment
        # self.alignment = 16

    def forward(self, x):
        # # Encoder
        # # -------------------------------------------
        #
        # x = relu(self.enc_conv0(input))  # enc_conv0 720
        #
        # x = relu(self.enc_conv1(x))      # enc_conv1 720
        # x = pool1 = pool(x)              # pool1
        #
        # x = relu(self.enc_conv2(x))      # enc_conv2 360
        # x = pool2 = pool(x)              # pool2
        #
        # x = relu(self.enc_conv3(x))      # enc_conv3 180
        # x = pool3 = pool(x)              # pool3
        #
        # x = relu(self.enc_conv4(x))      # enc_conv4 90
        # x = pool(x)                      # pool4
        #
        # # Bottleneck
        # x = relu(self.enc_conv5a(x))     # enc_conv5a 45
        # x = relu(self.enc_conv5b(x))     # enc_conv5b
        #
        # # Decoder
        # # -------------------------------------------
        #
        # x = upsample(x)                  # upsample4
        # x = concat(x, pool3)             # concat4
        # x = relu(self.dec_conv4a(x))     # dec_conv4a
        # x = relu(self.dec_conv4b(x))     # dec_conv4b
        #
        # x = upsample(x)                  # upsample3
        # x = concat(x, pool2)             # concat3
        # x = relu(self.dec_conv3a(x))     # dec_conv3a
        # x = relu(self.dec_conv3b(x))     # dec_conv3b
        #
        # x = upsample(x)                  # upsample2
        # x = concat(x, pool1)             # concat2
        # x = relu(self.dec_conv2a(x))     # dec_conv2a
        # x = relu(self.dec_conv2b(x))     # dec_conv2b
        #
        # x = upsample(x)                  # upsample1
        # x = concat(x, input)             # concat1
        # x = relu(self.dec_conv1a(x))     # dec_conv1a
        # x = relu(self.dec_conv1b(x))     # dec_conv1b
        #
        # x = self.dec_conv0(x)            # dec_conv0

        skips = []
        skips.append(x)
        for i in range(self.K):
            x = self.ds_path[i](x)
            x = self.pool(x, 2)
            skips.append(x)
        x = self.ds_path[self.K](x)
        skips.append(x)

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[2:]):
            x = torch.cat((F.interpolate(x, scale_factor=2), skip), 1)
            x = stage(x)

        return x

class model_SF(BaseModel):
    def __init__(self):
        super().__init__()
        self.filter = UNet_SF(23, 13)
        self.features = Features(transfer='log')

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.96)
        return [opt], [sched]

    def step(self, x, temporal):
        ''' temporal: precolor preoutput prefeature 13 '''
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
        predict = output[:3]

        return predict, output, grid

    def temporal_init(self, x):
        shape = list(x['reference'].shape)
        shape[1] = 13
        return torch.zeros(shape, dtype=x['reference'].dtype, device=x['reference'].device)

class model_SF_nppd(model_SF):
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.96)
        return [opt], [sched]

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
        predict = output[:3]

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
