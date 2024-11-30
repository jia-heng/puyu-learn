import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_transformer import SwinTransformerBlock
from .temporal_attention import TemporalAttention


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        '''
        input:
        x: (b, 9, w, h)
        return:
        (b, 9 + 23, w, h)
        '''
        shortcut = x
        x = self.extractor(x)
        return torch.cat((shortcut, x), dim=1)


class FeatureReweighting(nn.Module):
    def __init__(self):
        super(FeatureReweighting, self).__init__()

        self.reweighter = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, cur_raw_feats, prev_raw_feats, previous_features):
        '''
        input:
        current_features:  (b, 32, c, w, h)
        previous_features: (b, 32, c, w, h)
        '''
        input = torch.cat((cur_raw_feats, prev_raw_feats), dim=1)
        scores = self.reweighter(input)
        # scale to 0-10
        scores = (scores + 1) * 5.  # b, 4, w, h
        # scores = scores.unsqueeze(2)
        return scores * previous_features


class TemporalStabilization(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, input_feat=9, num_feat=32):
        super(TemporalStabilization, self).__init__()
        # temporal stabilization
        self.temporal_attn1 = nn.Conv2d(input_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(input_feat, num_feat, 3, 1, 1)

    def forward(self, cur_raw_feats, prev_raw_feats, x):
        """
        Args:
            cur_raw_feats (Tensor): current raw features with shape (b, 9, h, w).
            prev_raw_feats (Tensor): previous raw features with shape (b, 9, h, w).
            previous_features (Tensor): Aligned features with shape (b, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, c, h, w = x.size()
        cur_embedding = self.temporal_attn1(cur_raw_feats)
        prev_embedding = self.temporal_attn2(prev_raw_feats)
        corr = torch.sum(prev_embedding * cur_embedding, 1)  # (b, h, w)
        corr = corr.unsqueeze(1)  # (b, 1, h, w)

        corr_prob = torch.sigmoid(corr)  # (b, 1, h, w)
        corr_prob = corr_prob.expand(b, c, h, w)
        aligned_x = x * corr_prob

        return aligned_x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block(x)
        return out

    def flops(self, H, W):
        flops = H * W * self.in_channel * self.out_channel * (
                    3 * 3 + 1) + H * W * self.out_channel * self.out_channel * 3 * 3
        return flops


class DilationConv(nn.Module):
    def __init__(self, inchannel):
        super(DilationConv, self).__init__()
        self.dilation_1 = nn.Sequential(
            nn.Conv2d(inchannel, 64, 3, 1, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.dilation_2 = nn.Sequential(
            nn.Conv2d(inchannel, 64, 3, 1, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.dilation_5 = nn.Sequential(
            nn.Conv2d(inchannel, 64, 3, 1, 3, 3),
            nn.ReLU()
        )
        self.cat = nn.Sequential(
            nn.Conv2d(64 * 3, 64, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_1 = self.dilation_1(x)
        x_2 = self.dilation_2(x)
        x_5 = self.dilation_5(x)
        x = self.cat(torch.cat((x_1, x_2, x_5), dim=1))

        return x


# Upsample Block
class UpsamplePS(nn.Module):
    def __init__(self, in_channel, up_scale):
        super(UpsamplePS, self).__init__()
        self.in_channel = in_channel
        self.out_channel = in_channel * (2 ** up_scale)
        self.conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(upscale_factor=up_scale)

    def forward(self, x):
        out = self.ps(self.conv(x))
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


class ReconstructionNetwork(nn.Module):
    def __init__(self, in_channel):
        super(ReconstructionNetwork, self).__init__()
        self.encoder0 = SwinTransformerBlock(in_channel, num_heads=9)
        self.encoder1 = ConvBlock(in_channel, 64)
        self.encoder2 = ConvBlock(64, 32)

        self.encoder3 = ConvBlock(32, 64)
        self.encoder4 = ConvBlock(64, 64)

        self.mid1 = ConvBlock(64, 128)
        self.mid2 = ConvBlock(128, 128)

        self.decoder1 = ConvBlock(128 + 64, 64)
        self.decoder3 = ConvBlock(64 + 32, 64)

        self.ps1 = nn.PixelShuffle(2)
        self.ps2 = nn.PixelShuffle(2)
        self.conv_bf_ps1 = nn.Conv2d(128, 128 * 4, kernel_size=3, stride=1, padding=1)
        self.conv_bf_ps2 = nn.Conv2d(64, 64 * 4, kernel_size=3, stride=1, padding=1)

        self.decoder2 = ConvBlock(64, 64)
        self.decoder4 = ConvBlock(64, 4)

        self.sigmoid = nn.Sigmoid()

    def forward(self, current_features, acc_x):
        '''
        current_features: (b, c, w, h)
        previous_features: (b, c2, w, h)
        '''

        x = torch.cat((current_features, acc_x), dim=1)

        # encoder
        conv1 = self.encoder0(x)
        conv1 = self.encoder1(conv1)
        conv2 = self.encoder2(conv1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.encoder3(pool2)
        conv4 = self.encoder4(conv3)
        pool4 = F.max_pool2d(conv4, 2)

        # middle stage
        mid1 = self.mid1(pool4)
        mid2 = self.mid2(mid1)

        # decoder
        up5 = self.ps1(self.conv_bf_ps1(mid2))
        up5 = torch.cat((up5, conv4), dim=1)
        conv5 = self.decoder1(up5)

        conv6 = self.decoder2(conv5)
        up6 = self.ps2(self.conv_bf_ps2(conv6))
        up6 = torch.cat([up6, conv2], dim=1)

        conv7 = self.decoder3(up6)
        out = self.decoder4(conv7)

        x_t = out[:, :3, ...]
        alpha = self.sigmoid(out[:, 3:, ...])

        ret = alpha * x_t + (1. - alpha) * acc_x
        return ret


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.previous_est = None
        self.feature_extractor = FeatureExtractor()
        self.reconstruct = ReconstructionNetwork(45)
        # self.temporal_attention = TemporalStabilization()
        self.temporal_attention = TemporalAttention(ch=32, checkpoint=False)

    def warper2d(self, history, flow):
        h, w = history.size()[-2:]
        x_grid = torch.arange(0., w).to(history.device).float() + 0.5
        y_grid = torch.arange(0., h).to(history.device).float() + 0.5
        x_grid = (x_grid / w).view(1, 1, -1, 1).expand(1, h, -1, 1)  # 1, h, w, 1
        y_grid = (y_grid / h).view(1, -1, 1, 1).expand(1, -1, w, 1)  # 1, h, w, 1
        x_grid = x_grid * 2 - 1.
        y_grid = y_grid * 2 - 1.

        grid = torch.cat((x_grid, y_grid), dim=-1)  # b, h, w, 2
        flow = flow.permute(0, 2, 3, 1)  # b, h, w, 2

        grid = grid - flow * 2

        warped = F.grid_sample(history, grid, align_corners=True)
        return warped


    def forward(self, inputs, temporal):
        frame = inputs['image']
        mv = inputs['mv']
        depth = inputs['depth']
        normal = inputs['normal']
        albedo = inputs['albedo']

        x = torch.cat((frame, normal, depth, albedo), dim=1)

        prev_raw_feats = self.warper2d(temporal, mv)
        aligned_img = self.temporal_attention([prev_raw_feats[:, 0:3, ...], x, prev_raw_feats])

        cur_features = self.feature_extractor(x)
        denoised_img = self.reconstruct(cur_features, aligned_img)

        temporal = torch.cat([denoised_img, normal, depth, albedo], dim=1)

        return denoised_img, temporal

class model_ret_SSR(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.reconstruct = ReconstructionNetwork(45)
        self.temporal_attention = TemporalAttention(ch=32, checkpoint=False)
        # self.features = Features(transfer='log')

    def tensor_like(self, like, data):
        return torch.tensor(data, dtype=like.dtype, device=like.device)

    def temporal_init(self, x):
        shape = list(x['color'].shape)
        shape[1] = 10
        return torch.zeros(shape, dtype=x['color'].dtype, device=x['color'].device)

    def forward(self, color, depth, normal, albedo, motion, temporal):
        grid = self.create_meshgrid(motion)
        reprojected = F.grid_sample(
            temporal, # permute(0, 3, 1, 2)
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        encoder_feature = torch.concat((color, normal, albedo, depth), 1)
        aligned_img = self.temporal_attention([reprojected[:, 0:3, ...], encoder_feature, reprojected])
        # batch_size = color.shape[0]
        cur_features = self.feature_extractor(encoder_feature)
        denoised_img = self.reconstruct(cur_features, aligned_img)
        temporal = torch.cat([denoised_img, normal, depth, albedo], dim=1)

        return denoised_img, temporal


class model_ret_SSR(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.reconstruct = ReconstructionNetwork(45)
        self.temporal_attention = TemporalAttention(ch=32, checkpoint=False)
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(10, 32, 1),
        #     nn.LeakyReLU(0.3),
        #     nn.Conv2d(32, 32, 1),
        #     nn.LeakyReLU(0.3),
        #     nn.Conv2d(32, 32, 1)
        # )

        # self.filter = PartitioningPyramid()
        # self.weight_predictor = ConvUNet(
        #     70,
        #     self.filter.inputs
        # )
        self.features = Features(transfer='log')

    def temporal_init(self, x):
        shape = list(x['color'].shape)
        shape[1] = 10
        return torch.zeros(shape, dtype=x['color'].dtype, device=x['color'].device)

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
        ''' temporal: denoised img;
                    normal;
                    depth;
                    albedo;
                    '''
        grid = self.create_meshgrid(x['motion'])
        reprojected = F.grid_sample(
            temporal, # permute(0, 3, 1, 2)
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        # prev_color = reprojected[:, :3]
        # prev_output = reprojected[:, 3:6]
        # prev_feature = reprojected[:, 6:]
        color = x['color']
        normal = x['normal']
        albedo = x['albedo']
        depth = x['depth']
        encoder_feature = torch.concat((clip_logp1(normalize_radiance(color), normal, albedo, depth)), 1)
        aligned_img = self.temporal_attention([reprojected[:, 0:3, ...], encoder_feature, reprojected])
        # batch_size = color.shape[0]
        cur_features = self.feature_extractor(encoder_feature)
        denoised_img = self.reconstruct(cur_features, aligned_img)
        temporal = torch.cat([denoised_img, normal, depth, albedo], dim=1)

        return denoised_img, temporal, grid

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


class model_ret_SSR_nppd(model_ret_SSR):
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

        color = x['color']
        normal = x['normal']
        albedo = x['albedo']
        depth = x['depth']
        encoder_feature = torch.cat((clip_logp1(normalize_radiance(color), normal, albedo, depth)), 1)
        aligned_img = self.temporal_attention([reprojected[:, 0:3, ...], encoder_feature, reprojected])
        # batch_size = color.shape[0]
        cur_features = self.feature_extractor(encoder_feature)
        denoised_img = self.reconstruct(cur_features, aligned_img)
        temporal = torch.cat([denoised_img, normal, depth, albedo], dim=1)

        return denoised_img, temporal, grid

    def temporal_init(self, x):
        shape = list(x['reference'].shape)
        shape[1] = 10
        return torch.zeros(shape, dtype=x['reference'].dtype, device=x['reference'].device)
