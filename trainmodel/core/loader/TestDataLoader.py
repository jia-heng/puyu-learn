import torch
import pyexr
import os
import numpy as np
from .AugmentData import DataAugment

def walk_through_all_dirs(data_dir, data_name, gbuffers, spp):
    data_files = []
    name_path = os.path.join(data_dir + str(0), data_name)
    for name in sorted(os.listdir(name_path)):
        data_files.append({key + str(i): os.path.join(data_dir + str(i), value, value + name[-13:]) for key, value in gbuffers.items() for i in range(spp)})
    # random.shuffle(data_path_list)
    return data_files

class baseTestDataset(torch.utils.data.Dataset):
    def __init__(self, test, features, **kwargs):
        super().__init__()
        self.src = test
        self.augmentdata = DataAugment(self.src)
        data_path = test["path"]
        sequence_idxs = list(range(test["seqNum"] + test["sequenceBeginIdx"]))[test["sequenceBeginIdx"]:]
        self.files = list(map(lambda i: os.path.join(data_path, test["sequenceName"].format(index=i)), sequence_idxs))
        self.frames_per_sequence = test["frames"]
        self.features = features

    def get_feature_data(self, feature, channels, sequence_idx, frame_idx, sppNum=0):
        featurePath = os.path.join(
            self.files[sequence_idx],
            self.src["sppName"].format(index=sppNum),
            self.features[feature],
            self.src["featureName"].format(name=self.features[feature], index=frame_idx)
            )
        with pyexr.open(featurePath) as exr:
            data_np = exr.get()[:, :, :channels].astype(np.float32)
        return torch.from_numpy(data_np).permute(2, 0, 1)

    def __getitem__(self, idx):
        frame_idx = idx % self.frames_per_sequence  # color0000 % frame_idx
        sequence_idx = idx // self.frames_per_sequence  # scene0000 % sequence_idx

        temppath = os.path.join(self.files[sequence_idx], "spp00", "color")
        # 首帧的index
        index = int(sorted(os.listdir(temppath))[0].split('_')[1])
        data_idx = frame_idx + index

        # 定义特征及其通道数的字典
        buffers = {
            "color": 3,
            "diffuse": 3,
            "depth": 1,
            "normal": 3,
            "motionVector": 2,
            # "refraction": 1,
            # "reflection": 1,  # jihu 图层有问题，暂不添加
        }
        feature_data = {key: [] for key in buffers}
        for feature, channels in buffers.items():
            data = self.get_feature_data(feature, channels, sequence_idx, data_idx, 0)
            if feature in ["depth", "motionVector"]:
                data = torch.clamp(data, -5e3, 5e3)
            feature_data[feature].append(data)

        # 合并特征数据并计算平均值
        for feature in feature_data:
            stacked_tensors = torch.stack(feature_data[feature], dim=0)
            feature_data[feature] = torch.mean(stacked_tensors, dim=0)
        feature_data = self.augmentdata.apply_pad(feature_data)
        if frame_idx == 0:
            feature_data["motionVector"] = torch.zeros_like(feature_data["motionVector"])
        frame = {'color':   feature_data["color"],
                 'normal':  feature_data["normal"],
                 'depth':   torch.log(1 + 1/feature_data["depth"]),
                 'diffuse': feature_data["diffuse"],
                 # 'refraction': feature_data["refraction"],
                 # 'reflection': feature_data["reflection"],
                 'motion': feature_data["motionVector"],
                 # 'reference': feature_data["reference"],
                 'frame_index': torch.tensor(frame_idx),
                 'file': self.files[sequence_idx] + '_' + str(frame_idx)
                 }
        return frame

    def __len__(self):
        return self.src["seqNum"] * self.frames_per_sequence

    def savedir(self, path):
        return sum(len(n) for n in self.sequence)

    def to_torch(self, data, device):
        return {
            #key: torch.from_numpy(value).unsqueeze(0).to(device)
            key: value.to(device)
            #if isinstance(value, np.ndarray)
            if isinstance(value, torch.Tensor)
            else self.to_torch(value, device)
            if isinstance(value, dict)
            else value
            for key, value in data.items()
        }

class TestDataset_ME(baseTestDataset):
    def get_feature_data_ME(self, feature, channels, sequence_idx, frame_idx, sppNum=0):
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
        # idx: sequence_idx * frames_per_sequence + frame_idx
        frame_idx = idx % self.frames_per_sequence  # color0000 % frame_idx
        sequence_idx = idx // self.frames_per_sequence  # scene0000 % sequence_idx
        temppath = os.path.join(self.files[sequence_idx], "spp00", "color")
        # 首帧的index
        index = int(sorted(os.listdir(temppath))[0].split('_')[1][:-4])
        data_idx = frame_idx + index

        # 定义特征及其通道数的字典
        buffers = {
            "color": 3,
            "albedo": 3,
            "normal": 3,
            "depth": 1,
            "motionVector": 2,
            # "refraction": 1,
            # "reflection": 1,  # jihu 图层有问题，暂不添加
        }
        feature_data = {key: [] for key in buffers}

        for feature, channels in buffers.items():
            data = self.get_feature_data(feature, channels, sequence_idx, data_idx, 0)
            if feature in ["depth", "motionVector"]:
                data = torch.clamp(data, -5e3, 5e3)
            feature_data[feature].append(data)

        # 合并特征数据并计算平均值
        for feature in feature_data:
            stacked_tensors = torch.stack(feature_data[feature], dim=0)
            feature_data[feature] = torch.mean(stacked_tensors, dim=0)

        feature_data = self.augmentdata.apply_pad(feature_data)
        if frame_idx == 0:
            feature_data["motionVector"] = torch.zeros_like(feature_data["motionVector"])
        frame = {'color':   feature_data["color"],
                 'albedo': feature_data["albedo"],
                 'normal':  feature_data["normal"],
                 'depth':   feature_data["depth"],
                # 'refraction': feature_data["refraction"],
                # 'reflection': feature_data["reflection"],
                 'motion': feature_data["motionVector"],
                 # 'reference': feature_data["reference"],
                 'frame_index': torch.tensor(frame_idx),
                 'file': self.files[sequence_idx] + '_' + str(frame_idx)
                 }
        # frame['file'] = file TODO: load strings to pytorch
        # H W C Spp -> C H W Spp permute
        return frame

class TestDataset_nppd(baseTestDataset):
    def __getitem__(self, idx):
        frame_idx = idx % self.frames_per_sequence  # color0000 % frame_idx
        sequence_idx = idx // self.frames_per_sequence  # scene0000 % sequence_idx

        temppath = os.path.join(self.files[sequence_idx], "spp00", "color")
        # 首帧的index
        index = int(sorted(os.listdir(temppath))[0].split('_')[1])
        data_idx = frame_idx + index

        # 定义特征及其通道数的字典
        buffers = {
            "color": 3,
            "diffuse": 3,
            "depth": 1,
            "normal": 3,
            "motionVector": 2,
            # "refraction": 1,
            # "reflection": 1,  # jihu 图层有问题，暂不添加
        }
        feature_data = {key: [] for key in buffers}
        for feature, channels in buffers.items():
            data = self.get_feature_data(feature, channels, sequence_idx, data_idx, 0)
            if feature in ["depth", "motionVector"]:
                data = torch.clamp(data, -5e3, 5e3)
            feature_data[feature].append(data)

        # 合并特征数据并计算平均值
        for feature in feature_data:
            stacked_tensors = torch.stack(feature_data[feature], dim=0)
            feature_data[feature] = torch.mean(stacked_tensors, dim=0)
        feature_data = self.augmentdata.apply_pad(feature_data)
        if frame_idx == 0:
            feature_data["motionVector"] = torch.zeros_like(feature_data["motionVector"])
        frame = {'color':   feature_data["color"],
                 'normal':  feature_data["normal"],
                 'depth':   feature_data["depth"],
                 'diffuse': feature_data["diffuse"],
                 # 'refraction': feature_data["refraction"],
                 # 'reflection': feature_data["reflection"],
                 'motion': feature_data["motionVector"],
                 # 'reference': feature_data["reference"],
                 'frame_index': torch.tensor(frame_idx),
                 'file': self.files[sequence_idx] + '_' + str(frame_idx)
                 }
        return frame

class TestDataset_trt(torch.utils.data.Dataset):
    def __init__(self, test, features, **kwargs):
        super().__init__()
        self.src = test
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
            self.src["featureName"].format(name=self.features[feature], index=frame_idx)
            )
        with pyexr.open(featurePath) as exr:
            data_np = exr.get()[:, :, :channels].astype(np.float32)
            # data_np = exr.get(precision=pyexr.HALF)[:, :, :channels].astype(np.float16)
        return np.expand_dims(np.transpose(data_np, (2, 0, 1)), axis=0)

    def __getitem__(self, idx):
        frame_idx = idx % self.frames_per_sequence  # color0000 % frame_idx
        sequence_idx = idx // self.frames_per_sequence  # scene0000 % sequence_idx

        temppath = os.path.join(self.files[sequence_idx], "spp00", "color")
        # 首帧的index
        index = int(sorted(os.listdir(temppath))[0].split('_')[1])
        data_idx = frame_idx + index

        # 定义特征及其通道数的字典
        buffers = {
            "color": 3,
            "depth": 1,
            "normal": 3,
            "diffuse": 3,
            "motionVector": 2,
            # "refraction": 1,
            # "reflection": 1,  # jihu 图层有问题，暂不添加
        }

        feature_data = {}
        for feature, channels in buffers.items():
            data = self.get_feature_data(feature, channels, sequence_idx, data_idx, 0)
            # if feature in ["depth", "motionVector"]:
            #     data = torch.clamp(data, -5e3, 5e3)
            feature_data[feature] = data
        # feature_data = {key: [] for key in buffers}
        # for feature, channels in buffers.items():
        #     data = self.get_feature_data(feature, channels, sequence_idx, data_idx, 0)
        #     # if feature in ["depth", "motionVector"]:
        #     #     data = torch.clamp(data, -5e3, 5e3)
        #     feature_data[feature].append(data)

        # 合并特征数据并计算平均值
        # for feature in feature_data:
        #     stacked_tensors = torch.stack(feature_data[feature], dim=0)
        #     feature_data[feature] = torch.mean(stacked_tensors, dim=0)

        if frame_idx == 0:
            feature_data["motionVector"] = np.zeros_like(feature_data["motionVector"])
        frame = {'color':   feature_data["color"],
                 'depth':   np.log(1 + 1/feature_data["depth"]),
                 'normal':  feature_data["normal"],
                 'diffuse': feature_data["diffuse"],
                # 'refraction': feature_data["refraction"],
                # 'reflection': feature_data["reflection"],
                 'motion': feature_data["motionVector"],
                 # 'reference': feature_data["reference"],
                 # 'frame_index': torch.tensor(frame_idx),
                 # 'file': self.files[sequence_idx] + '_' + str(frame_idx)
                 }
        return frame

    def __len__(self):
        return self.src["seqNum"] * self.frames_per_sequence
