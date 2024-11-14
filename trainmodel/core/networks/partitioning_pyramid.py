import torch
import torch.nn as nn
import torch.nn.functional as F

def splat(img, kernel, size):
    h = img.shape[2]
    w = img.shape[3]
    total = torch.zeros_like(img)

    img = F.pad(img, [(size - 1) // 2] * 4)
    kernel = F.pad(kernel, [(size - 1) // 2] * 4)

    for i in range(size):
        for j in range(size):
            total += img[:, :, i:i+h, j:j+w] * kernel[:, i*size+j, None, i:i+h, j:j+w]
    
    return total

def upscale_quadrant(img, kernel, indices):
    # quad = torch.zeros(
    #     [img.shape[0], img.shape[1], img.shape[2] * 2, img.shape[3] * 2],
    #     dtype=img.dtype, device=img.device
    # )
    # quad[:, :, 0::2, 0::2] = img * kernel[:, indices[0], None, :, :]
    # quad[:, :, 0::2, 1::2] = img * kernel[:, indices[1], None, :, :]
    # quad[:, :, 1::2, 0::2] = img * kernel[:, indices[2], None, :, :]
    # quad[:, :, 1::2, 1::2] = img * kernel[:, indices[3], None, :, :]

    ps = nn.PixelShuffle(2)
    evev = img * kernel[:, indices[0], None, :, :]
    evod = img * kernel[:, indices[1], None, :, :]
    odev = img * kernel[:, indices[2], None, :, :]
    odod = img * kernel[:, indices[3], None, :, :]
    # temp = torch.cat((evev[:, 0:1], evod[:, 0:1], odev[:, 0:1], odod[:, 0:1],
    #                   evev[:, 1:2], evod[:, 1:2], odev[:, 1:2], odod[:, 1:2],
    #                   evev[:, 2:3], evod[:, 2:3], odev[:, 2:3], odod[:, 2:3]), dim=1)
    # quad1 = ps(temp)

    temp = torch.stack((evev, evod, odev, odod), dim=2).flatten(1, 2)
    quad = ps(temp)

    return quad

def upscale(img, kernel):
    img = F.pad(img, (1,1,1,1))
    kernel = F.pad(kernel, (1,1,1,1))

    tl = upscale_quadrant(img, kernel, [0, 1, 4, 5])
    tr = upscale_quadrant(img, kernel, [2, 3, 6, 7])
    bl = upscale_quadrant(img, kernel, [8, 9, 12, 13])
    br = upscale_quadrant(img, kernel, [10, 11, 14, 15])
    return tl[:,:,3:-1,3:-1] + tr[:,:,3:-1,1:-3] + bl[:,:,1:-3,3:-1] + br[:,:,1:-3,1:-3]

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

def splat_unfold(img, weight, size):
    # img shape: (N, C, H, W)
    # weight shape: (N, size*size, H, W)
    h = img.shape[2]
    w = img.shape[3]
    total = torch.zeros_like(img)

    img = F.pad(img, [(size - 1) // 2] * 4)
    weight = F.pad(weight, [(size - 1) // 2] * 4)

    for i in range(size):
        for j in range(size):
            total += img[:, :, i:i + h, j:j + w] * weight[:, i * size + j, None, i:i + h, j:j + w]

    return total

def upscale_pixelshuffle(img, kernel):
    ps = nn.PixelShuffle(2)

    # evev = torch.einsum('nchw, nchw->nchw', img, kernel[:, :3])
    # evod = torch.einsum('nchw, nchw->nchw', img, kernel[:, 3:6])
    # odev = torch.einsum('nchw, nchw->nchw', img, kernel[:, 6:9])
    # odod = torch.einsum('nchw, nchw->nchw', img, kernel[:, 9:12])

    evev = torch.einsum('nchw, nchw->nchw', img, kernel[:, 0, None]) + \
           torch.einsum('nchw, nchw->nchw', img, kernel[:, 1, None]) + \
           torch.einsum('nchw, nchw->nchw', img, kernel[:, 2, None])
    evod = torch.einsum('nchw, nchw->nchw', img, kernel[:, 3, None]) + \
           torch.einsum('nchw, nchw->nchw', img, kernel[:, 4, None]) + \
           torch.einsum('nchw, nchw->nchw', img, kernel[:, 5, None])
    odev = torch.einsum('nchw, nchw->nchw', img, kernel[:, 6, None]) + \
           torch.einsum('nchw, nchw->nchw', img, kernel[:, 7, None]) + \
           torch.einsum('nchw, nchw->nchw', img, kernel[:, 8, None])
    odod = torch.einsum('nchw, nchw->nchw', img, kernel[:, 9, None]) + \
           torch.einsum('nchw, nchw->nchw', img, kernel[:, 10, None]) + \
           torch.einsum('nchw, nchw->nchw', img, kernel[:, 11, None])

    temp = torch.cat((evev, evod, odev, odod), dim=1)
    return ps(temp)

class PartitioningPyramid_pixelshuffle():
    def __init__(self, K=5):
        self.K = K
        self.inputs = [25 + 25 + 1 + 1 + K] + [37 for i in range(K - 1)]
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
            denoised = denoised_levels[i] + upscale_pixelshuffle(denoised, F.softmax(weights[i + 1][:, 25:37], 1) * 4)

        previous = splat_unfold(previous, F.softmax(weights[0][:, 25:50], 1), 5)
        t_mu = torch.sigmoid(weights[0][:, 50, None])

        output = t_mu * previous + (1 - t_mu) * denoised

        return output