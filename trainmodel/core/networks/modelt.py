import torch
import torch.nn.functional as F
import torch.nn as nn
from ..util import clip_logp1, normalize_radiance, tensor_like
from .convunet import ConvUNet, ConvUNet_L, ConvUNet_S, ConvUNet_T
from .partitioning_pyramid import PartitioningPyramid, PartitioningPyramid_Large, PartitioningPyramid_Small
import lightning as L


class GrenderModel(L.LightningModule):
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

        # dims_and_depths=[(96, 96), (96, 128), (128, 192), (192, 256), (256, 384), (512, 512, 384)]
        self.weight_predictor = ConvUNet(
            70,
            self.filter.inputs #[57, 41, 41, 41, 41]
        )

        # self.features = Features(transfer='log')

    def create_meshgrid(self, motion):
        # x : {'color', 'depth', 'normal', 'diffuse', 'motion', 'reference', 'jitter', 'position'}
        batch_size, height, width, channel = motion.shape
        dtype = motion.dtype
        device = motion.device
        y = torch.arange(0, height, dtype=dtype, device=device)  # height
        x = torch.arange(0, width, dtype=dtype, device=device)  # width
        Y, X = torch.meshgrid(y, x, indexing='ij')
        meshgrid = torch.stack((X, Y), dim=-1)
        meshgrid = meshgrid.unsqueeze(0)
        meshgrid = meshgrid.repeat(batch_size, 1, 1, 1)
        # warp
        # motion vector 构造方式不同， 这里 加减 不同，grender 为 + - Me为 - -
        grid_0 = (meshgrid[:, :, :, 0] + motion[:, :, :, 0]).unsqueeze(3) / width * 2 - 1
        grid_1 = (meshgrid[:, :, :, 1] - motion[:, :, :, 1]).unsqueeze(3) / height * 2 - 1
        grid = torch.cat((grid_0, grid_1), 3)
        return grid

    def forward(self, color, depth, normal, diffuse, motion, temporal):
        # 将 step 方法的逻辑移到这里
        grid = self.create_meshgrid(motion.permute(0, 2, 3, 1))
        # reprojected = bilinear_grid_sample(temporal, grid, align_corners=False)
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

        encoder_input = torch.concat((
            depth,
            normal,
            diffuse,
            clip_logp1(normalize_radiance(color))
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
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature
        # predict = prev_output
        # self.filter = PartitioningPyramid()
        predict = self.filter(weights, color, prev_output)
        output2 = torch.concat((color, predict, feature), 1)
        return predict, output2

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

class model_kernel_L(model_kernel_init):
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
        # self.features = Features(transfer='log')

class model_kernel_S(model_kernel_init):
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
        # feature = self.encoder(torch.permute(encoder_input, (0, 4, 1, 2, 3)).flatten(0, 1))
        # feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1)
        # Denoiser

        weight_predictor_input = torch.concat((
            prev_color,
            color,
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
        # self.features = Features(transfer='log')
