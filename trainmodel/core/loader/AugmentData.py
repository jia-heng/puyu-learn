import torch.nn as nn
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F


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
        self.augCfg = {}
        # self.augCfg = self.set_augCfg(src, augCfg, sequence_idxs)

    def set_augCfg(self, sequence_idx, randomfunc):
        # maxSppNum = src["maxSppNum"]
        # cfg[sequence_idx]["spp"] = random.sample(range(maxSppNum), augCfg["sppNum"]) #从maxSppNum中随机取n个spp
        self.augCfg["crop"] = (randomfunc(sequence_idx, 0, self.height-self.crop + 1), randomfunc(sequence_idx, 0, self.width-self.crop + 1))
        self.augCfg["rotation"] = randomfunc(sequence_idx, 0, 4)
        self.augCfg["flip"] = randomfunc(sequence_idx, 0, 2)

    def to_gpu(self, frames):
        for key, value in frames.items():
            frames[key] = value.to("cuda:0")
        return frames

    def __iter__(self):
        # 同一个sequence执行相同操作
        pass

    def __call__(self, frames):
        # 同一个sequence执行相同操作
        augmentCfg = self.augCfg
        # reference
        for key, frame in frames.items():
            value = self.apply_crop(frame, augmentCfg)
            if augmentCfg["flip"] == 1:
                value = self.apply_flip(value)
            frames[key] = self.apply_rotation(value, augmentCfg["rotation"])
        return frames

    def apply_crop(self, frame, augmentCfg):
        """ 0:height_begin, 1:width_begin """
        C, H, W = frame.shape
        cropCfg = augmentCfg["crop"]
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
