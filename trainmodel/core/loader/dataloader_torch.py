import torch
import pyexr
import os
import numpy as np
# from .AugmentData import DataAugment
from multiprocessing.pool import ThreadPool
from .video_sampler import VideoSampler
from .misc import resolve_data_path, ACES, Shuffler
import lightning as L
import pyexr
from .AugmentData import DataAugment, FlipRotate
import zarr
from ..util import decompress_RGBE, screen_space_normal, motion_vectors, log_depth
import tqdm
def walk_through_all_dirs(data_dir, data_name, gbuffers, spp):
    data_files = []
    name_path = os.path.join(data_dir + str(0), data_name)
    for name in sorted(os.listdir(name_path)):
        data_files.append({key + str(i): os.path.join(data_dir + str(i), value, value + name[-13:]) for key, value in gbuffers.items() for i in range(spp)})
    # random.shuffle(data_path_list)
    return data_files

class TestDataset_torch(torch.utils.data.Dataset):
    def __init__(self, test, features, **kwargs):
        super().__init__()
        self.src = test
        # self.augmentdata = DataAugment(self.src)
        data_path = test["path"]
        sequence_idxs = list(range(test["seqNum"]))
        self.files = list(map(lambda i: os.path.join(data_path, test["sequenceName"].format(index=i)), sequence_idxs))
        self.frames_per_sequence = test["frames"]
        self.features = features

    def get_feature_data(self, feature, channels, sequence_idx, frame_idx, sppNum=0):
        featurePath = os.path.join(
            self.files[sequence_idx],
            self.src["sppName"].format(index=sppNum),
            self.features[feature],
            self.src["featureName"].format(index=frame_idx)
            )
        with pyexr.open(featurePath) as exr:
            data_np = exr.get()[:, :, :channels].astype(np.float32)
        return torch.from_numpy(data_np).permute(2, 0, 1)

    def __getitem__(self, idx):
        frame_idx = idx % self.frames_per_sequence  # color0000 % frame_idx
        sequence_idx = idx // self.frames_per_sequence  # scene0000 % sequence_idx

        temppath = os.path.join(self.files[sequence_idx], "spp00", "color")
        # 首帧的index
        index = int(sorted(os.listdir(temppath))[0].split('_')[1][:-4])
        data_idx = frame_idx + index

        # 定义特征及其通道数的字典
        buffers = {
            "color": 3,
            "depth": 1,
            "diffuse": 3,
            "normal": 3,
            "motionVector": 2
            # "refraction": 1,
            # "reflection": 1,  # jihu 图层有问题，暂不添加
        }
        feature_data = {key: [] for key in buffers}
        for feature, channels in buffers.items():
            data = self.get_feature_data(feature, channels, sequence_idx, data_idx, 0)
            # if feature in ["depth", "motionVector"]:
            #     data = torch.clamp(data, -5e3, 5e3)
            # if feature in ["depth"]:
            #     data[data == 0] = 5000
            feature_data[feature] = data

        # # 合并特征数据并计算平均值
        # for feature in feature_data:
        #     stacked_tensors = torch.stack(feature_data[feature], dim=0)
        #     feature_data[feature] = torch.mean(stacked_tensors, dim=0)
        # feature_data = self.augmentdata.apply_pad(feature_data)
        if frame_idx == 0:
            feature_data["motionVector"] = torch.zeros_like(feature_data["motionVector"])
        frame = {'color':   feature_data["color"],
                 'depth':   feature_data["depth"], #torch.log(1 + 1/feature_data["depth"]),
                 'normal': feature_data["normal"],
                 'diffuse': feature_data["diffuse"],
                # 'refraction': feature_data["refraction"],
                # 'reflection': feature_data["reflection"],
                 'motion': feature_data["motionVector"]
                 # 'reference': feature_data["reference"],
                 # 'frame_index': torch.tensor(frame_idx),
                 # 'file': self.files[sequence_idx] + '_' + str(frame_idx)
                 }
        return frame

    def __len__(self):
        return self.src["seqNum"] * self.frames_per_sequence

def normalize(v):
    """Individually normalize an array of vectors

    Args:
        v (ndarray, CHW): will be normalized along first dimension
    """
    return v / np.linalg.norm(v, axis=0, keepdims=True)

def compute_camera(target, up, pos):
    # simplified version of FlipRotate.apply_camera

    W = normalize(target - pos) # forward
    U = normalize(np.cross(W, up)) # right
    V = normalize(np.cross(U, W)) # up

    return W, V, U, pos,

class TestSampleDataset(torch.utils.data.Dataset):
    def __init__(self, test, features, **kwargs):
        super().__init__()
        self.files = list(map(lambda i: os.path.join(test['path'], test["frameName"].format(index=i)), range(test["frames"])))

        self.rendering_height = test['height']
        self.rendering_width = test['width']
        self.buffers = list(features.keys())
        self.samples = test["sppNum"]

    def __getitem__(self, idx):
        ds = zarr.group(store=zarr.ZipStore(self.files[idx], mode='r'))  # linux -> win 盘符前去掉 /

        forward, up, left, pos, = compute_camera(
            np.array(ds['camera_target']),
            np.array(ds['camera_up']),
            np.array(ds['camera_position']),
        )
        pv = np.array(ds['view_proj_mat'])

        frame = {
            'view_proj_mat': pv,
            'camera_position': pos,
            'camera_forward': forward,
            'camera_up': up,
            'camera_left': left,
            'crop_offset': np.array([28, 0], dtype=np.int32),
        }

        if idx > 0:
            pds = zarr.group(store=zarr.ZipStore(self.files[idx - 1], mode='r'))
            prev_forward, prev_up, prev_left, prev_pos = compute_camera(
                np.array(pds['camera_target']),
                np.array(pds['camera_up']),
                np.array(pds['camera_position']),
            )
            prev_pv = np.array(pds['view_proj_mat'])

            frame['prev_camera'] = {
                'view_proj_mat': prev_pv,
                'camera_position': prev_pos,
                'camera_forward': prev_forward,
                'camera_up': prev_up,
                'camera_left': prev_left,
                'crop_offset': np.array([28, 0], dtype=np.int32),
            }
            pds.store.close()
        else:
            frame['prev_camera'] = frame.copy()

        frame['frame_index'] = np.array((idx,), dtype=np.int32),
        # frame['file'] = file TODO: load strings to pytorch

        if 'w_normal' in self.buffers or 'normal' in self.buffers:
            w_normal = ds['normal'][..., 28:-28, :, 0:self.samples].astype(np.float32)

        if 'w_normal' in self.buffers:
            frame['w_normal'] = w_normal

        if 'normal' in self.buffers:
            frame['normal'] = screen_space_normal(w_normal, forward, up, left)

        if 'depth' in self.buffers or 'motion' in self.buffers or 'w_position' in self.buffers:
            w_position = ds['position'][..., 28:-28, :, 0:self.samples]

        if 'motion' in self.buffers or 'w_motion' in self.buffers:
            w_motion = ds['motion'][..., 28:-28, :, 0:self.samples]

        if 'w_position' in self.buffers:
            frame['w_position'] = w_position

        if idx > 0:
            if 'w_motion' in self.buffers:
                frame['w_motion'] = w_motion

            if 'motion' in self.buffers:
                motion = motion_vectors(
                    w_position, w_motion,
                    pv, prev_pv,
                    self.rendering_height, self.rendering_width
                )
                frame['motion'] = np.clip(motion, -5e3, 5e3)
        else:
            if 'w_motion' in self.buffers:
                frame['w_motion'] = np.zeros_like(w_motion)

            if 'motion' in self.buffers:
                frame['motion'] = np.zeros_like(w_motion[0:2])

        if 'depth' in self.buffers:
            frame['depth'] = log_depth(w_position, pos)

        if 'color' in self.buffers:
            exposure = np.array(ds['exposure'])
            rgbe = ds['color'][..., 28:-28, :, 0:self.samples]
            frame['color'] = decompress_RGBE(rgbe, exposure)

        if 'diffuse' in self.buffers:
            frame['diffuse'] = ds['diffuse'][..., 28:-28, :, 0:self.samples].astype(np.float32)

        frame['reference'] = np.array(ds['reference'][:, 28:-28, :])
        if self.samples > 1:
            for key in self.buffers:
                frame[key] = np.mean(frame[key], axis=-1)

        ds.store.close()

        return frame

    def __len__(self):
        return len(self.files)
