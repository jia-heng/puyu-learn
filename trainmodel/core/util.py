import torch
import torch.distributed as dist
import pyexr
import os
import numpy as np

def tensor_like(like, data):
    return torch.tensor(data, dtype=like.dtype, device=like.device)

###################
# Tonemapping
###################
def lrange(start, stop, step=1):
    return list(range(start, stop, step))

def normalize_radiance(luminance, return_mean = False):
    mean = torch.mean(luminance, lrange(1, luminance.dim()), keepdim=True) + 1e-8

    if return_mean:
        return luminance / mean, mean
    else:
        return luminance / mean


def clip_logp1(x):
    return torch.log(torch.maximum(x, torch.zeros_like(x)) + 1)

###################
# Distributed
###################


def dist_cat(tensor_in):
    if dist.is_available() and dist.is_initialized():
        shape = list(tensor_in.shape)
        shape[0] *= dist.get_world_size()
        tensor_out = torch.zeros(shape, dtype=tensor_in.dtype, device=tensor_in.device)
        dist.all_gather_into_tensor(tensor_out, tensor_in)
        return tensor_out
    else:
        return tensor_in


def rank_zero():
    return dist.get_rank() == 0


def decompress_RGBE(color, exposures):
    """Decompresses per-sample radiance from RGBE compressed data

    Args:
        color (ndarray, uint8, 4HWS): radiance data in RGBE representation
        [min_exposure, max_exposure]: exposure range for decompression

    Returns:
        color (ndarray, 3HWS): per-sample RGB radiance
    """
    exponents = (color.astype(np.float32)[3] + 1) / 256
    # exposures = np.reshape(exposures, (1, 1, 1, 2))

    exponents = np.exp(exponents * (exposures[1] - exposures[0]) + exposures[0])
    color = color.astype(np.float32)[:3] / 255 * exponents[np.newaxis]
    return color


def compress_RGBE(color):
    """Computes RGBE compressed representation of radiance data

    Args:
        color (ndarray, 3HWS): per-sample RGB radiance

    Returns:
        color (ndarray, uint8, 4HWS): radiance data in RGBE representation
        [min_exposure, max_exposure]: exposure range for decompression
    """
    log_radiance = np.log(color[np.where(color > 0)])

    if log_radiance.size == 0:  # Handle black frames
        return np.zeros((4, color.shape[1], color.shape[2], color.shape[3]), dtype=np.uint8), [0, 0]

    # Calculate exposure
    min_exp = np.min(log_radiance)
    max_exp = np.max(log_radiance)

    # Get exponent from brightest channel
    brightest_channel = np.max(color, axis=0)
    exponent = np.ones_like(brightest_channel) * -np.inf
    np.log(brightest_channel, out=exponent, where=brightest_channel > 0)

    # Quantise exponent with ceiling function
    e_channel = np.minimum((exponent - min_exp) / (max_exp - min_exp) * 256, 255).astype(np.uint8)[np.newaxis]
    # Actually encoded exponent
    exponent = np.exp(((e_channel.astype(np.float32) + 1) / 256) * (max_exp - min_exp) + min_exp)

    # Quantise colour channels
    rgb_float = (color / exponent) * 255
    rgb_channels = (rgb_float).astype(np.uint8)
    # Add dither (exponents were quantised with ceiling so this doesn't go over 255)
    rgb_channels += ((rgb_float - rgb_channels) > np.random.random(rgb_channels.shape))

    return np.concatenate([rgb_channels, e_channel]), [min_exp, max_exp]


def save_exr_with_path(file: object, id: object, save_path: object = r'midi\temp') -> object:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for key, tensor in file.items():
        if isinstance(tensor, torch.Tensor):
            # 将PyTorch张量转换为numpy数组
            tensor = tensor.squeeze(0)
            if (len(tensor.shape) == 3):
                tensor = tensor.permute(1, 2, 0)
                pyexr.write(os.path.join(save_path, key + str(id) + '.exr'), tensor.numpy().astype('float16'))
            if (len(tensor.shape) == 4):
                tensor = tensor.permute(1, 2, 0, 3)
                for i in range(tensor.shape[-1]):
                    pyexr.write(os.path.join(save_path, key + str(id) + '_' + str(i) + '.exr'), tensor[...,i].numpy().astype('float16'))
        else:
            print(f"Value for key '{key}' is not a PyTorch tensor and will not be saved.")
