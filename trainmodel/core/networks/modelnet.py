import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convunet import ConvUNet
from .partitioning_pyramid import PartitioningPyramid
from ..loss.loss import Features, SMAPE, BaseLoss
from ..util import normalize_radiance, clip_logp1, dist_cat
import os
import pyexr
import numpy as np
# def st(t):
#     return np.transpose(t.cpu().numpy(), (1, 2, 0))

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

        grid_0 = (meshgrid[:, :, 0] + motion[:, 0])
        grid_1 = (meshgrid[:, :, 1] - motion[:, 1])
        grid = torch.stack((grid_0, grid_1), -1) + 0.5

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
            x['diffuse'],
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
            loss = None
                # loss = torch.tensor(0.0, requires_grad=True)
        else:
            y_1, temporal, _ = self.step(self.prev_x, self.temporal)
            y_2, _, grid = self.step(x, temporal)
            loss, y = self.two_step_loss(
                torch.stack((self.prev_x['reference'], x['reference']), 1),
                torch.stack((y_1, y_2), 1),
                grid
            )

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

        if x['frame_index'] == 63:
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

    def save_images(self, images, batch_idx, epoch):
        for i, image in enumerate(images):
            self.logger.experiment.add_image(f'denoised/{batch_idx}-{i}', image, epoch)

class model_ME(BaseModel):

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.96)
        return [opt], [sched]

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

class model_nppd(BaseModel):
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

    def step(self, x, temporal):
        grid = self.create_meshgrid(x['motion'].permute(0, 2, 3, 1))
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

        # Sample encoder

        color = x['color']
        batch_size = color.shape[0]
        encoder_input = torch.concat((
            x['depth'],
            x['normal'],
            x['diffuse'],
            # x['refraction'],
            # x['reflection'],
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

class model_vgg(BaseModel):
    def __init__(self):
        super().__init__()
        self.build_loss()
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

    def build_loss(self):
        self.loss_utils = BaseLoss()
        self.loss_utils.to_gpu("cuda")

    def vgg_step(self, x):
        if x['frame_index'] == 0:
            temporal = self.temporal_init(x)
            y, temporal, _ = self.step(x, temporal)
        else:
            y, temporal, _ = self.step(x, self.temporal)
        self.temporal = temporal.detach()
        batch_loss_dict = self.loss_utils.calc_total_loss(y, x['reference'])
        loss = batch_loss_dict["total_loss"]

        return loss, y

    def training_step(self, x, batch_idx):
        loss, y = self.vgg_step(x)
        # batch_loss_dict["total_loss"].backward()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, x, batch_idx):
        loss, y = self.vgg_step(x)

        if x['frame_index'] == 10:
            y = normalize_radiance(y)
            y = torch.pow(y / (y + 1), 1 / 2.2)
            y = dist_cat(y).cpu().numpy()

            # Writing images can block the rank 0 process for a couple seconds
            # which often breaks distributed training so we start a separate thread
            Thread(target=self.save_images, args=(y, batch_idx, self.trainer.current_epoch)).start()
            self.log("val_loss", loss, on_epoch=True)
            return loss

        self.log("val_loss", loss, on_epoch=True)
        return loss
