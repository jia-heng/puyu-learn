import torch.nn as nn
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

def normalize(v):
    """Individually normalize an array of vectors

    Args:
        v (ndarray, CHW): will be normalized along first dimension
    """
    return v / np.linalg.norm(v, axis=0, keepdims=True)

class DataAugment:
    """ C H W """
    def __init__(self, src, augment=False, augCfg=None, sequence_idxs=None):
        self.height = src["height"]
        self.width = src["width"]
        offset_H = 0
        if self.height % 32 != 0:
            offset_H = (self.height // 32 + 1) * 32 - self.height
        offset_W = 0
        if self.width % 32 != 0:
            offset_W = (self.width // 32 + 1) * 32 - self.width
        self.padding = (0, offset_W, 0, offset_H)

        if not augment:
            return
        self.crop = augCfg["crop"]

    def to_gpu(self, frames):
        for key, value in frames.items():
            frames[key] = value.to("cuda:0")
        return frames

    def __iter__(self):
        # 同一个sequence执行相同操作
        pass

    def __call__(self, frames, sequence_idx, randomfunc):
        # 同一个sequence执行相同操作
        offset = (randomfunc(sequence_idx, 0, self.height-self.crop + 1), randomfunc(sequence_idx, 0, self.width-self.crop + 1))
        rotation = randomfunc(sequence_idx, 0, 4)
        flip = randomfunc(sequence_idx, 0, 1)
        # reference
        for key, frame in frames.items():
            value = self.apply_crop(frame, offset)
            if flip == 1:
                value = self.apply_flip(value)
            frames[key] = self.apply_rotation(value, float(rotation))
        return frames

    def apply_crop(self, frame, cropCfg):
        """ 0:height_begin, 1:width_begin """
        C, H, W = frame.shape
        H_beg = cropCfg[0]
        W_beg = cropCfg[1]
        size = self.crop
        return frame[:, H_beg:H_beg + size, W_beg:W_beg + size]

    def apply_downSample(self, frame):
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        ret = avg_pool(frame)
        return ret

    def apply_pad(self, frames):
        """ C H W """
        for key, value in frames.items():
            frames[key] = F.pad(value.unsqueeze(0), self.padding, mode='reflect').squeeze(0)
        return frames

    def apply_flip(self, frame):
        ret = frame.flip(2)  # Flip horizontally
        return ret

    def apply_rotation(self, frame, orientation):
        if orientation == 0:
            return frame
        return TF.rotate(frame, 90 * orientation)

class FlipRotate:
    """Class representing a flip/rotation data augmentation

    After initialization the transformation can be applied to many cameras and arrays
    """

    def __init__(
            self,
            orientation,
            height,
            width,
            window
    ):
        """
        Args:
            orientation (int): [0,7] representing the 8 possible axis-aligned flips and rotations. 0 is identity.
            height (int): height of the camera resolution (in pixels)
            width (int): width of the camera resolution (in pixels)
            window (int): height and width of the cropped image
        """
        self.orientation = orientation
        self.height = height
        self.width = width
        self.window = window

    def apply_camera(self, target, up, pos, p, offset):
        """Applies orientation change to camera intrinsics

        Args:
            target (ndarray, size (3)): a world-space point the camera is pointing at (center of the frame)
            up (ndarray, size (3)): vector in world-space that points upward in screen-space
            pos (ndarray, size (3)): the camera's position in world-space
            p (ndarray, size (4,4)): projective matrix of the camera e.g.
                [0.984375, 0.,   0.,      0.     ],
                [0.,       1.75, 0.,      0.     ],
                [0.,       0.,   1.0001, -0.10001],
                [0.,       0.,   1.,      0.     ]
            offset (ndarray, size (2)): offset of random crop (window) from top left corner of camera frame (in pixels)

        Returns:
            W (ndarray, size (3)): vector in world-space that points forward in screen-space
            V (ndarray, size (3)): vector in world-space that points up in screen-space
            U (ndarray, size (3)): vector in world-space that points right in screen-space
            pos (ndarray, size (3)): unchanged camera position
            offset (ndarray, size (2)): transformed offset, MAY BE NEGATIVE!
            pv (ndarray, size (4,4)): computed view-projection matrix, ROW VECTOR
        """

        # make orthonormal camera basis
        W = normalize(target - pos)  # forward
        U = normalize(np.cross(W, up))  # right
        V = normalize(np.cross(U, W))  # up

        # flip rotate offset and basis
        if self.orientation % 2 < 1:
            U = -U
            offset[1] = self.width - offset[1] - self.window

        if self.orientation % 4 < 2:
            V = -V
            offset[0] = self.height - offset[0] - self.window

        if self.orientation % 8 < 4:
            U, V = V, U
            offset = (self.height + self.width) // 2 - self.window - np.flip(offset)

        # view matrix for transformed camera basis
        view_basis = np.pad(np.stack([U, V, W], 0), [[0, 1], [0, 1]]) + np.diag([0., 0., 0., 1.])

        # view matrix for camera position
        view_translate = np.pad(-pos[:, np.newaxis], [[0, 1], [3, 0]]) + np.diag([1., 1., 1., 1.])

        # combined view matrix
        v = np.matmul(view_basis, view_translate)

        # view-projection matrix
        # DirectX ROW VECTOR ALERT!
        pv = np.matmul(v.T, p.T).astype(np.float32)

        return W, V, U, pos, offset, pv

    def apply_array(self, x):
        """Applies orientation change to per-pixel data

        Args:
            x (ndarray, CHW...): will be transformed, may have additional final dimensions
        """
        if self.orientation % 2 < 1:
            x = np.flip(x, 2)

        if self.orientation % 4 < 2:
            x = np.flip(x, 1)

        if self.orientation % 8 < 4:
            x = np.flip(np.transpose(x, [0, 2, 1] + list(range(3, x.ndim))), (1, 2))

        return np.ascontiguousarray(x)
