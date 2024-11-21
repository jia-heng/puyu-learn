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

def screen_space_normal(w_normal, W, V, U):
    """Transforms per-sample world-space normals to screen-space / relative to camera direction

    Args:
        w_normal (ndarray, 3HWS): per-sample world-space normals
        W (ndarray, size (3)): vector in world-space that points forward in screen-space
        V (ndarray, size (3)): vector in world-space that points up in screen-space
        U (ndarray, size (3)): vector in world-space that points right in screen-space

    Returns:
        normal (ndarray, 3HWS): per-sample screen-space normals
    """
    # TODO: support any number of extra dimensions like apply_array
    return np.einsum('ij, ihws -> jhws', np.stack([W, U, V], axis=1), w_normal)  # column vectors

def screen_space_position(w_position, pv, height, width):
    """Projects per-sample world-space positions to screen-space (pixel coordinates)

    Args:
        w_normal (ndarray, 3HWS): per-sample world-space positions
        pv (ndarray, size (4,4)): camera view-projection matrix
        height (int): height of the camera resolution (in pixels)
        width (int): width of the camera resolution (in pixels)

    Returns:
        projected (ndarray, 2HWS): Per-sample screen-space position (pixel coordinates).
            IJ INDEXING! for gather ops and consistency,
            see backproject_pixel_centers in noisebase.torch.projective for use with grid_sample.
            Degenerate positions give inf.
    """
    # TODO: support any number of extra dimensions like apply_array
    homogeneous = np.concatenate((  # Pad to homogeneous coordinates
        w_position,
        np.ones_like(w_position)[0:1]
    ))

    # ROW VECTOR ALERT!
    # DirectX uses row vectors...
    projected = np.einsum('ij, ihws -> jhws', pv, homogeneous)
    projected = np.divide(
        projected[0:2], projected[3],
        out=np.zeros_like(projected[0:2]),
        where=projected[3] != 0
    )

    # directx pixel coordinate fluff
    projected = projected * np.reshape([0.5 * width, -0.5 * height], (2, 1, 1, 1)).astype(np.float32) \
                + np.reshape([width / 2, height / 2], (2, 1, 1, 1)).astype(np.float32)

    projected = np.flip(projected, 0)  # height, width; ij indexing

    return projected

def motion_vectors(w_position, w_motion, pv, prev_pv, height, width):
    """Computes per-sample screen-space motion vectors (in pixels)

    Args:
        w_position (ndarray, 3HWS): per-sample world-space positions
        w_motion (ndarray, 3HWS): per-sample world-space positions
        pv (ndarray, size (4,4)): camera view-projection matrix
        prev_pv (ndarray, size (4,4)): camera view-projection matrix from previous frame
        height (int): height of the camera resolution (in pixels)
        width (int): width of the camera resolution (in pixels)

    Returns:
        motion (ndarray, 2HWS): Per-sample screen-space motion vectors (in pixels).
            IJ INDEXING! for gather ops and consistency,
            see backproject_pixel_centers in noisebase.torch.projective for use with grid_sample.
            Degenerate positions give inf.
    """
    # TODO: support any number of extra dimensions like apply_array (only the docstring here)
    current = screen_space_position(w_position, pv, height, width)
    prev = screen_space_position(w_position + w_motion, prev_pv, height, width)

    motion = prev - current

    return motion

def log_depth(w_position, pos):
    """Computes per-sample compressed depth (disparity-ish)

    Args:
        w_position (ndarray, 3HWS): per-sample world-space positions
        pos (ndarray, size (3)): the camera's position in world-space

    Returns:
        motion (ndarray, 1HWS): per-sample compressed depth
    """
    # TODO: support any number of extra dimensions like apply_array
    d = np.linalg.norm(w_position - np.reshape(pos, (3, 1, 1, 1)), axis=0, keepdims=True)
    return np.log(1 + 1 / d)

def backproject_pixel_centers(motion, crop_offset, prev_crop_offset, as_grid = False):
    """Decompresses per-sample radiance from RGBE compressed data

    Args:
        motion (tensor, N2HW): Per-sample screen-space motion vectors (in pixels)
            see `noisebase.projective.motion_vectors`
        crop_offset (tensor, size (2)): offset of random crop (window) from top left corner of camera frame (in pixels)
        prev_crop_offset (tensor, size (2)): offset of random crop (window) in previous frame
        as_grid (bool): torch.grid_sample, with align_corners = False format

    Returns:
        pixel_position (tensor, N2HW): ij indexed pixel coordinates OR
        pixel_position (tensor, NHW2): xy WH position (-1, 1) IF as_grid
    """
    height = motion.shape[2]
    width = motion.shape[3]
    dtype = motion.dtype
    device = motion.device

    pixel_grid = torch.stack(torch.meshgrid(
        torch.arange(0, height, dtype=dtype, device=device),
        torch.arange(0, width, dtype=dtype, device=device),
        indexing='ij'
    ))

    pixel_pos = pixel_grid + motion - prev_crop_offset[..., None, None] + crop_offset[..., None, None]

    if as_grid:
        # as needed for grid_sample, with align_corners = False
        pixel_pos_xy = torch.permute(torch.flip(pixel_pos, (1,)), (0, 2, 3, 1)) + 0.5
        image_pos = pixel_pos_xy / tensor_like(pixel_pos, [width, height])
        return image_pos * 2 - 1
    else:
        return pixel_pos
