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
        self.augCfg = self.set_augCfg(src, augCfg, sequence_idxs)


    def set_augCfg(self, src, augCfg, sequence_idxs):
        maxSppNum = src["maxSppNum"]
        height = src["height"]
        width = src["width"]
        cfg = {key: {} for key in sequence_idxs}
        for sequence_idx in sequence_idxs:
            cfg[sequence_idx]["spp"] = random.sample(range(maxSppNum), augCfg["sppNum"])
            cfg[sequence_idx]["downsize"] = random.randrange(2)
            cfg[sequence_idx]["crop"] = (random.randrange(height-self.crop + 1), random.randrange(width-self.crop + 1))
            cfg[sequence_idx]["rotation"] = random.randrange(4)
            cfg[sequence_idx]["flip"] = random.randrange(2)
        return cfg

    def to_gpu(self, frames):
        for key, value in frames.items():
            frames[key] = value.to("cuda:0")
        return frames

    def __iter__(self):
        # 同一个sequence执行相同操作
        pass

    def __call__(self, frames, sequenceIdx):
        # 同一个sequence执行相同操作
        augmentCfg = self.augCfg[sequenceIdx]
        # reference
        for key, frame in frames.items():
            if augmentCfg["downsize"] == 1:
                frame = self.apply_downSample(frame)
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
        if augmentCfg["downsize"] == 1:
            H_beg = int(H_beg / (2 * H - size) * (H - size))
            W_beg = int(W_beg / (2 * W - size) * (W - size))
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
