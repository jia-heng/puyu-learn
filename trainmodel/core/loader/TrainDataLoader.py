import torch
import os
import numpy as np
from multiprocessing.pool import ThreadPool
from .video_sampler import VideoSampler
from .misc import resolve_data_path, ACES, Shuffler
import lightning as L
import pyexr
from .AugmentData import DataAugment, FlipRotate
import zarr
from ..util import decompress_RGBE, screen_space_normal, motion_vectors, log_depth

import torch.nn.functional as F
import imageio.v3 as iio

class TrainingSampleDataset(torch.utils.data.Dataset):
    def __init__(self, src, sequence_idxs, rng, augment, batch_size, augm, features, **kwargs):
        super().__init__()
        self.src = src
        data_path = src["path"]
        data_path = resolve_data_path(data_path)
        self.sequence_idxs = sequence_idxs
        self.files = list(map(lambda i: os.path.join(data_path, src["sequenceName"].format(index=i)), sequence_idxs))
        self.frames_per_sequence = src["frames"]
        self.rng = rng
        self.augment = augment
        self.features = features
        self.batch_size = batch_size  # Just for worker_init_fn
        self.augm = augm

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
        # idx: sequence_idx * frames_per_sequence + frame_idx
        frame_idx = idx['idx'] % self.frames_per_sequence  # color0000 % frame_idx
        sequence_idx = idx['idx'] // self.frames_per_sequence  # scene0000 % sequence_idx
        ref_path = os.path.join(self.files[sequence_idx], "reference")
        # 首帧的index
        index = int(sorted(os.listdir(ref_path))[0].split('_')[1])
        data_idx = frame_idx + index
        with pyexr.open(os.path.join(ref_path, self.src["referenceName"].format(index=data_idx))) as exr:
            reference = exr.get()[:, :, :3].astype(np.float32)
        reference = torch.from_numpy(reference).permute(2, 0, 1)

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

        # 数据增强 - 随机 spp
        augmentdata = self.augm
        randomSppIdxs = augmentdata.augCfg[self.sequence_idxs[sequence_idx]]["spp"] if self.augment else range(self.src["maxSppNum"])
        for n in randomSppIdxs:
            for feature, channels in buffers.items():
                data = self.get_feature_data(feature, channels, sequence_idx, data_idx, n)
                if feature in ["depth", "motionVector"]:
                    data = torch.clamp(data, -5e3, 5e3)
                feature_data[feature].append(data)

        # 合并特征数据并计算平均值
        for feature in feature_data:
            stacked_tensors = torch.stack(feature_data[feature], dim=0)
            feature_data[feature] = torch.mean(stacked_tensors, dim=0)

        # reference = augmentdata.apply_downSample(reference)
        feature_data["reference"] = reference
        # feature_data = augmentdata.to_gpu(feature_data)
        # 数据增强 crop - flip - rotation
        if self.augment:
            feature_data = augmentdata(feature_data, self.sequence_idxs[sequence_idx])
        else:
            feature_data = augmentdata.apply_pad(feature_data)
        if frame_idx == 0:
            feature_data["motionVector"] = torch.zeros_like(feature_data["motionVector"])
        frame = {'color':   feature_data["color"],
                 'normal':  feature_data["normal"],
                 'depth':   torch.log(1 + 1/feature_data["depth"]),
                 'diffuse': feature_data["diffuse"],
                # 'refraction': feature_data["refraction"],
                # 'reflection': feature_data["reflection"],
                 'motion': feature_data["motionVector"],
                 'reference': feature_data["reference"],
                 'frame_index': torch.tensor(frame_idx),
                 'file': self.files[sequence_idx] + '_' + str(frame_idx)
                 }
        # frame['file'] = file TODO: load strings to pytorch
        # H W C Spp -> C H W Spp permute
        return frame

    def __getitems__(self, idxs):
        if hasattr(self, 'thread_pool'):
            return self.thread_pool.map(self.__getitem__, idxs)
        else:
            return list(map(self.__getitem__, idxs))

class TrainingSampleDataset_ME(TrainingSampleDataset):

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

    def save_exr(self, idx, image, path, imgType):
        # image = np.transpose(image.cpu().numpy()[0], (1,2,0))
        # image = (ACES(image)*255).astype(np.uint8)
        # self.save_pool.apply_async(iio.imwrite, [file, image])
        imgType = str(imgType)
        output_path = os.path.join(path, 'exr')
        os.makedirs(output_path, exist_ok=True)
        filename = '{imgType}_{idx:04d}.exr'.format(imgType=imgType, idx=idx)
        file_path = os.path.join(output_path, filename)
        image_array = np.transpose(image.cpu().numpy(), (1, 2, 0))
        pyexr.write(file_path, image_array)

    def save_png(self, idx, image, path, imgType):
        imgType = str(imgType)
        output_path = os.path.join(path, 'png')
        os.makedirs(output_path, exist_ok=True)
        filename = '{imgType}_{idx:04d}.png'.format(imgType=imgType, idx=idx)
        file_path = os.path.join(output_path, filename)

        image = np.transpose(image.cpu().numpy(), (1, 2, 0))
        image = (ACES(image) * 255).astype(np.uint8)
        iio.imwrite(file_path, image)

    def __getitem__(self, idx):
        # idx: sequence_idx * frames_per_sequence + frame_idx
        frame_idx = idx['idx'] % self.frames_per_sequence  # color0000 % frame_idx
        sequence_idx = idx['idx'] // self.frames_per_sequence  # sequence_idxs 的索引, 不一定是真实值
        ref_path = os.path.join(self.files[sequence_idx], "reference")
        # 首帧的index
        index = int(sorted(os.listdir(ref_path))[0].split('_')[1][:-4])
        data_idx = frame_idx + index
        with pyexr.open(os.path.join(ref_path, self.src["referenceName"].format(index=data_idx))) as exr:
            reference = exr.get()[:, :, :3].astype(np.float32)
        reference = torch.from_numpy(reference).permute(2, 0, 1)

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

        # randomSppIdxs = augmentdata.augCfg[self.sequence_idxs[sequence_idx]]["spp"] if self.augment else range(self.src["maxSppNum"])
        for n in range(self.src["maxSppNum"]):
            for feature, channels in buffers.items():
                data = self.get_feature_data_ME(feature, channels, sequence_idx, data_idx, n)
                if feature in ["depth", "motionVector"]:
                    data = torch.clamp(data, -5e3, 5e3)
                feature_data[feature].append(data)

        # 合并特征数据并计算平均值
        for feature in feature_data:
            stacked_tensors = torch.stack(feature_data[feature], dim=0)
            feature_data[feature] = torch.mean(stacked_tensors, dim=0)

        reference = self.augm.apply_downSample(reference)
        feature_data["reference"] = reference
        # feature_data = self.augm.to_gpu(feature_data)
        # 数据增强 crop - flip - rotation
        if self.augment:
            feature_data = self.augm(feature_data, self.sequence_idxs[sequence_idx], self.rng.derive(idx['epoch']).integers)
        else:
            feature_data = self.augm.apply_pad(feature_data)
        # path = r'.\test\temp'
        # for key,value in feature_data.items():
        # self.save_png(idx['idx'], feature_data['color'], path, 'color')
        # if self.augment:
        #     self.save_png(idx['idx'], feature_data['reference'], path, 'reference')
        # self.save_png(idx['idx'], F.pad(feature_data['motionVector'], (0, 0, 0, 0, 0, 1)), path, 'motionVector')

        if frame_idx == 0:
            feature_data["motionVector"] = torch.zeros_like(feature_data["motionVector"])
        frame = {'color':   feature_data["color"],
                 'albedo': feature_data["albedo"],
                 'normal':  feature_data["normal"],
                 'depth':   feature_data["depth"],
                # 'refraction': feature_data["refraction"],
                # 'reflection': feature_data["reflection"],
                 'motion': feature_data["motionVector"],
                 'reference': feature_data["reference"],
                 'frame_index': torch.tensor(frame_idx),
                 'file': self.files[sequence_idx] + '_' + str(frame_idx)
                 }
        # frame['file'] = file TODO: load strings to pytorch
        # H W C Spp -> C H W Spp permute
        return frame

class TrainingSampleDataset_nppd(torch.utils.data.Dataset):
    def __init__(self, src, sequence_idxs, rng, augment, batch_size, features, ** kwargs):
        super().__init__()
        data_path = resolve_data_path(src["path"])
        self.files = list(map(lambda i: os.path.join(data_path, src["sequenceName"].format(index=i)), sequence_idxs))
        self.frames_per_sequence = src['frames']
        self.rng = rng
        self.augment = augment
        self.crop = src['crop']
        self.rendering_height = src['height']
        self.rendering_width = src['width']
        self.buffers = list(features.keys())
        self.samples = src["sppNum"]
        self.batch_size = batch_size # Just for worker_init_fn

    def __getitem__(self, idx):
        # idx: sequence_idx * frames_per_sequence + frame_idx
        frame_idx = idx['idx'] % self.frames_per_sequence  # 3
        sequence_idx = idx['idx'] // self.frames_per_sequence  # 2

        if self.augment:
            orientation = self.rng.derive(idx['epoch']).integers(sequence_idx, 0, 8)
        else:
            orientation = 7
        flip_rotate = FlipRotate(orientation, self.rendering_height, self.rendering_width, self.crop)

        # ds = zarr.group(store = zarr.ZipStore(self.files[sequence_idx][1:], mode='r')) #linux
        ds = zarr.group(store=zarr.ZipStore(self.files[sequence_idx], mode='r'))

        forward, up, left, pos, offset, pv = flip_rotate.apply_camera(
            ds['camera_target'][frame_idx],
            ds['camera_up'][frame_idx],
            ds['camera_position'][frame_idx],
            ds['proj_mat'][frame_idx],
            ds['crop_offset'][frame_idx]
        )

        frame = {
            'view_proj_mat': pv,
            'camera_position': pos,
            'camera_forward': forward,
            'camera_up': up,
            'camera_left': left,
            'crop_offset': offset,
        }

        if frame_idx > 0:
            prev_forward, prev_up, prev_left, prev_pos, prev_offset, prev_pv = flip_rotate.apply_camera(
                ds['camera_target'][frame_idx - 1],
                ds['camera_up'][frame_idx - 1],
                ds['camera_position'][frame_idx - 1],
                ds['proj_mat'][frame_idx - 1],
                ds['crop_offset'][frame_idx - 1]
            )
            frame['prev_camera'] = {
                'view_proj_mat': prev_pv,
                'camera_position': prev_pos,
                'camera_forward': prev_forward,
                'camera_up': prev_up,
                'camera_left': prev_left,
                'crop_offset': prev_offset,
            }
        else:
            frame['prev_camera'] = frame.copy()

        frame['frame_index'] = frame_idx,
        frame['file'] = self.files[sequence_idx]

        if 'w_normal' in self.buffers or 'normal' in self.buffers:
            w_normal = flip_rotate.apply_array(ds['normal'][frame_idx, ..., 0:self.samples]).astype(np.float32)

        if 'w_normal' in self.buffers:
            frame['w_normal'] = w_normal

        if 'normal' in self.buffers:
            frame['normal'] = screen_space_normal(w_normal, forward, up, left)

        if 'depth' in self.buffers or 'motion' in self.buffers or 'w_position' in self.buffers:
            w_position = flip_rotate.apply_array(ds['position'][frame_idx, ..., 0:self.samples])

        if 'motion' in self.buffers or 'w_motion' in self.buffers:
            w_motion = flip_rotate.apply_array(ds['motion'][frame_idx, ..., 0:self.samples])

        if 'w_position' in self.buffers:
            frame['w_position'] = w_position

        if frame_idx > 0:
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
            exposure = ds['exposure'][frame_idx]
            rgbe = flip_rotate.apply_array(ds['color'][frame_idx, ..., 0:self.samples])
            frame['color'] = decompress_RGBE(rgbe, exposure)

        if 'diffuse' in self.buffers:
            frame['diffuse'] = flip_rotate.apply_array(ds['diffuse'][frame_idx, ..., 0:self.samples]).astype(np.float32)

        if 'reference' in self.buffers:
            frame['reference'] = flip_rotate.apply_array(ds['reference'][frame_idx])

        # for key in self.buffers:
        #     frame[key] = torch.mean(frame[key], dim=-1)

        ds.store.close()

        return frame

def collate_fixes(batch):
    # batch_size = len(batch)
    collated = torch.utils.data.default_collate(batch)
    collated['frame_index'] = collated['frame_index'][0]
    return collated

def collate_fixes_ME(batch):
    # batch_size = len(batch)
    collated = torch.utils.data.default_collate(batch)
    collated['frame_index'] = collated['frame_index'][0][0]
    return collated

def worker_init_fn(worker_id):
    # random.seed(worker_id * 40 + current_epoch)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # In the main process
        return
    # Loader pool
    worker_info.dataset.thread_pool = ThreadPool(worker_info.dataset.batch_size)

class MeTrainingSampleLoader(torch.utils.data.DataLoader):
    def __init__(self, training_data, config, get_epoch=None, **kwargs):
        # RNG for shuffling, test-train splitting, augmentations
        rng = Shuffler(config["seed"])
        stage = config["stage"]  # 'train' or 'val'
        shuffle = config["shuffle"]
        batch_size = config["batch_size"]
        num_workers = config["num_workers"]
        augment = config["augment"]  # Augmentation
        sequence_idxs = np.arange(training_data["seqNum"])

        val_idxs, train_idxs = rng.split(-1, sequence_idxs, config["val_split"], shuffle, batch_size)
        sequence_idxs = train_idxs if stage == 'train' else val_idxs

        sampler = VideoSampler(
            batch_size,  # Sampler computes correct distributed batch size
            training_data["frames"],
            len(sequence_idxs),
            shuffle=shuffle and stage == 'train',
            shuffle_fn=lambda epoch, sequence: rng.shuffle(epoch, sequence),
            drop_last=config["drop_last"],
            get_epoch=get_epoch
        )

        augm = DataAugment(
            src=training_data,
            augment=config["augment"] and stage == 'train',
            augCfg=config["augmentCfg"],
            sequence_idxs=sequence_idxs
        )

        ds = TrainingSampleDataset_ME(
            src=training_data,
            sequence_idxs=sequence_idxs,
            rng=rng,
            augment=config["augment"] and stage == 'train',
            batch_size=sampler.batch_size,
            augm=augm,
            **kwargs
        )

        super().__init__(
            ds,
            batch_sampler=sampler,
            collate_fn=collate_fixes,
            num_workers=num_workers,
            # num_workers=0,
            worker_init_fn=worker_init_fn,
            prefetch_factor=4 if num_workers > 0 else None,
            # multiprocessing_context='spawn', # Fork may break with CUDA, but spawn starts very slowly
            pin_memory=True  # Speeed
        )

    ### Resumable data loader

    def state_dict(self):
        return self.batch_sampler.cached_state

    def load_state_dict(self, state_dict):
        if state_dict != {}:
            self.batch_sampler.start_epoch_idx = state_dict['start_epoch_idx']
            self.batch_sampler.epoch_idx = state_dict['start_epoch_idx']
            self.batch_sampler.start_sequence_idx = state_dict['start_sequence_idx']

class NppdTrainingSampleLoader(torch.utils.data.DataLoader):
    def __init__(self, training_data, config, get_epoch=None, **kwargs):
        # RNG for shuffling, test-train splitting, augmentations
        rng = Shuffler(config["seed"])
        stage = config["stage"]  # 'train' or 'val'
        shuffle = config["shuffle"]
        batch_size = config["batch_size"]
        num_workers = config["num_workers"]
        sequence_idxs = np.arange(training_data["seqNum"])
        val_idxs, train_idxs = rng.split(-1, sequence_idxs, config["val_split"], shuffle, batch_size)
        sequence_idxs = train_idxs if stage == 'train' else val_idxs

        sampler = VideoSampler(
            batch_size,  # Sampler computes correct distributed batch size
            training_data["frames"],
            len(sequence_idxs),
            shuffle=shuffle and stage == 'train',
            shuffle_fn=lambda epoch, sequence: rng.shuffle(epoch, sequence),
            drop_last=config["drop_last"],
            get_epoch=get_epoch
        )

        ds = TrainingSampleDataset_nppd(
            src=training_data,
            sequence_idxs=sequence_idxs,
            rng=rng,
            augment=config["augment"] and stage == 'train',
            batch_size=sampler.batch_size,
            **kwargs
        )

        super().__init__(
            ds,
            batch_sampler=sampler,
            collate_fn=collate_fixes_ME,
            num_workers=num_workers,
            # num_workers=0,
            worker_init_fn=worker_init_fn,
            prefetch_factor=4 if num_workers > 0 else None,
            # multiprocessing_context='spawn', # Fork may break with CUDA, but spawn starts very slowly
            pin_memory=True  # Speeed
        )

    def state_dict(self):
        return self.batch_sampler.cached_state

    def load_state_dict(self, state_dict):
        if state_dict != {}:
            self.batch_sampler.start_epoch_idx = state_dict['start_epoch_idx']
            self.batch_sampler.epoch_idx = state_dict['start_epoch_idx']
            self.batch_sampler.start_sequence_idx = state_dict['start_sequence_idx']

class TrainingSampleLoader_L_ME(L.LightningDataModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.loader_args = config

    def setup(self, stage):
        # We don't want pytorch lightning to mess with the sampler
        self.trainer._accelerator_connector.use_distributed_sampler = False

    def train_dataloader(self):
        self.loader_args["config"]["stage"] = "train"
        return MeTrainingSampleLoader(get_epoch=lambda: self.trainer.current_epoch, **self.loader_args)

    def val_dataloader(self):
        self.loader_args["config"]["stage"] = "val"
        return MeTrainingSampleLoader(get_epoch=lambda: self.trainer.current_epoch, **self.loader_args)

class TrainingSampleLoader_L_NPPD(L.LightningDataModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.loader_args = config

    def setup(self, stage):
        # We don't want pytorch lightning to mess with the sampler
        self.trainer._accelerator_connector.use_distributed_sampler = False

    def train_dataloader(self):
        self.loader_args["config"]["stage"] = "train"
        return NppdTrainingSampleLoader(get_epoch=lambda: self.trainer.current_epoch, **self.loader_args)

    def val_dataloader(self):
        self.loader_args["config"]["stage"] = "val"
        return NppdTrainingSampleLoader(get_epoch=lambda: self.trainer.current_epoch, **self.loader_args)
