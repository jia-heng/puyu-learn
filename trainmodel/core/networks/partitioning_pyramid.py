import torch
import torch.nn as nn
import torch.nn.functional as F

def splat(img, kernel, size):
    h = img.shape[2]
    w = img.shape[3]
    total = torch.zeros_like(img)

    img = F.pad(img, [size // 2] * 4)
    kernel = F.pad(kernel, [size // 2] * 4)

    for i in range(size):
        for j in range(size):
            total += img[:, :, i:i+h, j:j+w] * kernel[:, i*size+j, None, i:i+h, j:j+w]
    
    return total

def upscale_quadrant(img, kernel, indices):
    # ps = nn.PixelShuffle(2)
    evev = img * kernel[:, indices[0], None, :, :]
    evod = img * kernel[:, indices[1], None, :, :]
    odev = img * kernel[:, indices[2], None, :, :]
    odod = img * kernel[:, indices[3], None, :, :]

    temp = torch.stack((evev, evod, odev, odod), dim=2).flatten(1, 2)
    # quad = ps(temp)

    return F.pixel_shuffle(temp, upscale_factor=2)

def upscale(img, kernel):
    img = F.pad(img, (1,1,1,1))
    kernel = F.pad(kernel, (1,1,1,1))

    tl = upscale_quadrant(img, kernel, [0, 1, 4, 5])
    tr = upscale_quadrant(img, kernel, [2, 3, 6, 7])
    bl = upscale_quadrant(img, kernel, [8, 9, 12, 13])
    br = upscale_quadrant(img, kernel, [10, 11, 14, 15])
    return tl[:,:,3:-1,3:-1] + tr[:,:,3:-1,1:-3] + bl[:,:,1:-3,3:-1] + br[:,:,1:-3,1:-3]

def upscale_v1(img, kernel, size=4):
    # img = F.pad(img, (1,1,1,1))
    # kernel = F.pad(kernel, (1,1,1,1))
    # idxs = [[3, -1], [1, -3]]
    # total = 0
    # for i in range(size):
    #     idx0 = i // 2
    #     idx1 = i % 2
    #     total += F.pixel_shuffle(
    #         torch.cat([
    #             img * kernel[:, 4 * i, None],
    #             img * kernel[:, 4 * i + 1, None],
    #             img * kernel[:, 4 * i + 2, None],
    #             img * kernel[:, 4 * i + 3, None]
    #         ], dim=1),
    #         2)[:, :, idxs[idx0][0]:idxs[idx0][1], idxs[idx1][0]:idxs[idx1][1]]
    # return total
    tl = upscale_quadrant(img, kernel, [0, 1, 4, 5])
    tr = upscale_quadrant(img, kernel, [2, 3, 6, 7])
    bl = upscale_quadrant(img, kernel, [8, 9, 12, 13])
    br = upscale_quadrant(img, kernel, [10, 11, 14, 15])
    return tl + tr + bl + br

def upscale_v2(img, kernel, size=4):
    evev = evod = odev = odod = 0
    for i in range(size):
        evev += img * kernel[:, 4*i, None]
        evod += img * kernel[:, 4*i+1, None]
        odev += img * kernel[:, 4*i+2, None]
        odod += img * kernel[:, 4*i+3, None]
    temp = torch.cat([evev, evod, odev, odod], dim=1)

    return F.pixel_shuffle(temp, upscale_factor=2)

def splat_unfold(img, weight, size):
    # img shape: (N, C, H, W)
    # weight shape: (N, size*size, H, W)
    B, C, H, W = img.shape
    padding = size // 2
    total = torch.sum(
        F.unfold(img, size, padding=padding).view(B, C, size*size, H, W) *
        weight.view(B, 1, size*size, H, W),
        dim=2
    )

    return total

class PartitioningPyramid():
    def __init__(self, K = 5):
        self.K = K
        self.inputs = [25 + 25 + 1 + 1 + K] + [41 for i in range(K-1)]
        self.t_lambda_index = 51
    
    def __call__(self, weights, rendered, previous):   

        part_weights = F.softmax(weights[0][:, 52:], 1)
        partitions = part_weights[:, :, None] * rendered[:, None]

        denoised_levels = [
            splat(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:25], 1),
                5
            )
            for i in range(self.K)
        ]

        denoised = denoised_levels[-1]
        for i in reversed(range(self.K - 1)):
            denoised = denoised_levels[i] + upscale(denoised, F.softmax(weights[i+1][:, 25:41], 1) * 4)

        previous = splat(previous, F.softmax(weights[0][:, 25:50], 1), 5)
        t_mu = torch.sigmoid(weights[0][:, 50, None])

        output = t_mu * previous + (1 - t_mu) * denoised

        return output

def upscale_Large_v1(img, kernel):
    img = F.pad(img, (1,1,1,1))
    kernel = F.pad(kernel, (1,1,1,1))

    tl = upscale_quadrant(img, kernel, [0, 1, 4, 5])
    tr = upscale_quadrant(img, kernel, [2, 3, 6, 7])
    return tl[:,:,3:-1,3:-1] + tr[:,:,1:-3,1:-3]
    # return tl + tr

def upscale_Large(img, kernel):
    ps = nn.PixelShuffle(2)
    temp = torch.cat([
        img * kernel[:, 0, None, :, :] + img * kernel[:, 4, None, :, :],  # evev
        img * kernel[:, 1, None, :, :] + img * kernel[:, 5, None, :, :],  # evod
        img * kernel[:, 2, None, :, :] + img * kernel[:, 6, None, :, :],  # odev
        img * kernel[:, 3, None, :, :] + img * kernel[:, 7, None, :, :]  # odod
    ], dim=1)
    quad = ps(temp)

    return quad

class PartitioningPyramid_Large():
    def __init__(self, K=5):
        self.K = K
        self.inputs = [16 + 16 + 1 + 1 + K] + [24 for i in range(K - 1)]
        self.t_lambda_index = 33

    def __call__(self, weights, rendered, previous):
        part_weights = F.softmax(weights[0][:, 34:], 1)
        partitions = part_weights[:, :, None] * rendered[:, None]

        denoised_levels = [
            splat(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:16], 1),
                4
            )
            for i in range(self.K)
        ]

        denoised = denoised_levels[-1]
        for i in reversed(range(self.K - 1)):
            denoised = denoised_levels[i] + upscale_Large(denoised, F.softmax(weights[i + 1][:, 16:24], 1) * 4)

        previous = splat(previous, F.softmax(weights[0][:, 16:32], 1), 4)
        t_mu = torch.sigmoid(weights[0][:, 32, None])

        output = t_mu * previous + (1 - t_mu) * denoised

        return output

def upscale_Small(img, kernel):
    ps = nn.PixelShuffle(2)
    temp = torch.cat([
        img * kernel[:, 0, None, :, :],  # evev
        img * kernel[:, 1, None, :, :],  # evod
        img * kernel[:, 2, None, :, :],  # odev
        img * kernel[:, 3, None, :, :]  # odod
    ], dim=1)
    quad = ps(temp)

    return quad

class PartitioningPyramid_Small():
    def __init__(self, K=5):
        self.K = K
        self.inputs = [9 + 9 + 1 + 1 + K] + [13 for i in range(K - 1)]
        self.t_lambda_index = 19
    def __call__(self, weights, rendered, previous):
        part_weights = F.softmax(weights[0][:, 20:], 1)
        partitions = part_weights[:, :, None] * rendered[:, None]

        denoised_levels = [
            splat_unfold(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:9], 1),
                3
            )
            for i in range(self.K)
        ]

        denoised = denoised_levels[-1]
        for i in reversed(range(self.K - 1)):
            denoised = denoised_levels[i] + upscale_Small(denoised, F.softmax(weights[i + 1][:, 9:13], 1) * 4)

        previous = splat_unfold(previous, F.softmax(weights[0][:, 9:18], 1), 3)
        t_mu = torch.sigmoid(weights[0][:, 18, None])

        output = t_mu * previous + (1 - t_mu) * denoised

        return output

class PartitioningPyramid_Small_part(PartitioningPyramid_Small):
    def __init__(self, K=5):
        self.K = K
        self.inputs = [9 + 9 + 1 + 1] + [13 for i in range(K - 1)]
        self.t_lambda_index = 19

    def __call__(self, weights, rendered, previous):
        denoised_levels = [
            splat_unfold(
                F.avg_pool2d(rendered, 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:9], 1),
                3
            )
            for i in range(self.K)
        ]

        denoised = denoised_levels[-1]
        for i in reversed(range(self.K - 1)):
            denoised = denoised_levels[i] + upscale_Small(denoised, F.softmax(weights[i + 1][:, 9:13], 1) * 4)

        previous = splat_unfold(previous, F.softmax(weights[0][:, 9:18], 1), 3)
        t_mu = torch.sigmoid(weights[0][:, 18, None])

        output = t_mu * previous + (1 - t_mu) * denoised

        return output

def splat_NPSF(img, kernel, size):
    h = img.shape[2]
    w = img.shape[3]
    total = torch.zeros_like(img)

    img = F.pad(img, [size // 2] * 4)
    kernel = F.pad(kernel, [size // 2] * 4)

    for i in range(size):
        for j in range(size):
            total += img[:, :, i:i + h, j:j + w] * kernel[:, i * size + j, None, i:i + h, j:j + w]

    return total

class PartitioningPyramid_NPSF():
    def __init__(self, K=5):
        self.K = K
        self.inputs = [25 + 25 + 1 + 1 + K] + [41 for i in range(K - 1)]
        self.t_lambda_index = 51

    def __call__(self, weights, rendered, previous):
        part_weights = F.softmax(weights[0][:, 52:], 1)
        partitions = part_weights[:, :, None] * rendered[:, None]

        denoised_levels = [
            splat_unfold(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:25], 1),
                5
            )
            for i in range(self.K)
        ]

        denoised = denoised_levels[-1]
        for i in reversed(range(self.K - 1)):
            denoised = denoised_levels[i] + upscale(denoised, F.softmax(weights[i + 1][:, 25:41], 1) * 4)

        previous = splat(previous, F.softmax(weights[0][:, 25:50], 1), 5)
        t_mu = torch.sigmoid(weights[0][:, 50, None])

        output = t_mu * previous + (1 - t_mu) * denoised

        return output

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
        # dims_and_depths = [
        #     (64, 64),
        #     (96),
        #     (128),
        #     (192, 256),
        #     (256, 384),
        #     (512, 512, 384)
        # ]
        dims_and_depths = [
            (32, 32),
            (32, 64),
            (64, 96),
            (96, 160),
            (160, 256),
            (384, 384, 256)
        ]
        self.ds_path = nn.ModuleList()
        prev_dim = in_channels
        for dims in dims_and_depths:
            layers = []
            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))

        # self.ps_path = nn.ModuleList()
        self.us_path = nn.ModuleList()
        for dims in reversedl(dims_and_depths)[1:]:
            # self.ps_path.append(nn.Conv2d(dims[-1], dims[-1] * 4, 1))
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[-1], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]
            for dim in reversedl(dims):
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim
            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(out_chans, dims_and_depths[:-1] + [reversedl(dims_and_depths[-1])]):
            self.out_path.append(nn.Conv2d(dims[0], out_chan, 1))
    def upscale_bil(self, x):
        return F.upsample(x)
    def forward(self, x, prev_color, color, previous, prev_feature, feature):
        skips = []
        for i in range(self.K):
            x = self.ds_path[i](x)
            skips.append(x)
            x = self.pool(x, 2)
        x = self.ds_path[self.K](x)
        skips.append(x)

        x = skips[-1]
        i = 0
        j = self.K - i - 1
        x = torch.cat((F.interpolate(x, scale_factor=2), skips[j]), 1)
        # x = torch.cat((self.ps(self.ps_path[i](x)), skips[j]), 1)
        x = self.us_path[i](x)
        weight = self.out_path[j](x)
        t_lambda = weight[:, self.t_lambda_index, None]
        rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
        partition = weight[:, self.partial, None] * rendering
        denoising = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3)
        denoising_us = upscale_Small(denoising, F.softmax(weight[:, 11:15], 1) * 4)

        i = 1
        j = self.K - i - 1
        x = torch.cat((F.interpolate(x, scale_factor=2), skips[j]), 1)
        # x = torch.cat((self.ps(self.ps_path[i](x)), skips[j]), 1)
        x = self.us_path[i](x)
        weight = self.out_path[j](x)
        t_lambda = weight[:, self.t_lambda_index, None]
        rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
        partition = weight[:, self.partial, None] * rendering
        denoising = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3) + denoising_us
        denoising_us = upscale_Small(denoising, F.softmax(weight[:, 11:15], 1) * 4)

        i = 2
        j = self.K - i - 1
        x = torch.cat((F.interpolate(x, scale_factor=2), skips[j]), 1)
        # x = torch.cat((self.ps(self.ps_path[i](x)), skips[j]), 1)
        x = self.us_path[i](x)
        weight = self.out_path[j](x)
        t_lambda = weight[:, self.t_lambda_index, None]
        rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
        partition = weight[:, self.partial, None] * rendering
        denoising = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3) + denoising_us
        denoising_us = upscale_Small(denoising, F.softmax(weight[:, 11:15], 1) * 4)

        i = 3
        j = self.K - i - 1
        x = torch.cat((F.interpolate(x, scale_factor=2), skips[j]), 1)
        # x = torch.cat((self.ps(self.ps_path[i](x)), skips[j]), 1)
        x = self.us_path[i](x)
        weight = self.out_path[j](x)
        t_lambda = weight[:, self.t_lambda_index, None]
        rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
        partition = weight[:, self.partial, None] * rendering
        denoising = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3) + denoising_us
        denoising_us = upscale_Small(denoising, F.softmax(weight[:, 11:15], 1) * 4)

        i = 4
        j = self.K - i - 1
        x = torch.cat((F.interpolate(x, scale_factor=2), skips[j]), 1)
        # x = torch.cat((self.ps(self.ps_path[i](x)), skips[j]), 1)
        x = self.us_path[i](x)
        weight = self.out_path[j](x)
        t_lambda = weight[:, self.t_lambda_index, None]
        rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
        partition = weight[:, self.partial, None] * rendering
        denoised = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3) + denoising_us

        previous = splat_unfold(previous, F.softmax(weight[:, -9:], 1), 3)
        t_mu = torch.sigmoid(weight[:, self.pre_lambda_index, None])
        predict = t_mu * previous + (1 - t_mu) * denoised
        color_ac = rendering
        feature_ac = t_lambda * prev_feature + (1 - t_lambda) * feature

        return predict, color_ac, feature_ac

        # if len(self.out_path) == len(self.ds_path):
        #     outputs.append(self.out_path[-1](x))
        # for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
        #     x = torch.cat((self.ps(self.ps_path[i](x)), skip), 1)
        #     x = stage(x)
        #     if i < self.K:
        #         weight = self.out_path[i](x)
        #         j = self.K - i - 1
        #         t_lambda = weight[:, self.t_lambda_index, None]
        #         rendering = t_lambda * F.avg_pool2d(prev_color, 2 ** j, 2 ** j) + (1 - t_lambda) * F.avg_pool2d(color, 2 ** j, 2 ** j)
        #         partition = weight[:, self.partial, None] * rendering
        #         denoising = splat_unfold(partition, F.softmax(weight[:, 0:9], 1), 3)
        #         output = output + upscale_Small(denoised, F.softmax(weight[:, 11:15], 1) * 4)
        #
        # previous = splat_unfold(previous, F.softmax(weights[0][:, 9:18], 1), 3)
        # t_mu = torch.sigmoid(weights[0][:, 18, None])
        #
        # predict = t_mu * previous + (1 - t_mu) * denoised
        # return reversedl(outputs)

