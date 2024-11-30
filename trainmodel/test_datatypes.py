import numpy as np
from core.networks.modelnet import BaseModel, model_nppd, model_ME
from core.networks.modelMe import model_kernel_S, model_kernel_T
import torch
import os
import glob
import re
import pyexr
from core.loader.TestDataLoader import baseTestDataset, TestDataset_nppd, TestDataset_ME, TestDataset_ME_new
import json
from core.loader.misc import ACES
import torch.nn as nn
import torch.nn.functional as F
import imageio.v3 as iio

def toTorch(data, device):
    return {
        # key: torch.from_numpy(value).unsqueeze(0).to(device)
        key: value.to(device)
        # if isinstance(value, np.ndarray)
        if isinstance(value, torch.Tensor)
        else toTorch(value, device)
        if isinstance(value, dict)
        else value
        for key, value in data.items()
    }

def save_png(idx, image, path, imgType):
    imgType = str(imgType)
    output_path = os.path.join(path, 'png')
    os.makedirs(output_path, exist_ok=True)
    filename = '{imgType}_{idx:04d}.exr'.format(imgType=imgType, idx=idx)
    file_path = os.path.join(output_path, filename)

    image = np.transpose(image.numpy(), (1,2,0))
    image = (ACES(image)*255).astype(np.uint8)
    iio.imwrite(file_path, image)

def save_exr(idx, image, path, imgType):
    imgType = str(imgType)
    output_path = os.path.join(path, 'exr')
    os.makedirs(output_path, exist_ok=True)
    filename = '{imgType}_{idx:04d}.exr'.format(imgType=imgType, idx=idx)
    file_path = os.path.join(output_path, filename)
    image_array = np.transpose(image.numpy(), (1, 2, 0))
    pyexr.write(file_path, image_array)

def save_exr_psnr(idx, image, path, imgType, psnr):
    imgType = str(imgType)
    output_path = os.path.join(path, 'exr')
    os.makedirs(output_path, exist_ok=True)
    filename = '{imgType}_{idx:04d}'.format(imgType=imgType, idx=idx) + '_' + psnr + '.exr'
    file_path = os.path.join(output_path, filename)
    image_array = np.transpose(image.numpy(), (1, 2, 0))
    pyexr.write(file_path, image_array)

def calc_psnr(predict, gt):
    v1 = torch.clip(predict, 0.0, 1.0)
    v2 = torch.clip(gt, 0.0, 1.0)
    mse = F.mse_loss(v1, v2)
    res = 10 * torch.log10(1.0 / (mse + 0.00001))
    return float(res)

def calcPSNR(img, idx, config_info):
    test = config_info["test"]
    data_path = test["path"]
    sequence_idxs = list(range(test["seqNum"] + test["sequenceBeginIdx"]))[test["sequenceBeginIdx"]:]
    files = list(map(lambda i: os.path.join(data_path, test["sequenceName"].format(index=i)), sequence_idxs))
    frame_idx = idx % 64  # color0000 % frame_idx
    sequence_idx = idx // 64  # sequence_idxs 的索引, 不一定是真实值
    referencepath = os.path.join(files[sequence_idx], "reference")
    # 首帧的index
    index = int(sorted(os.listdir(referencepath))[0].split('_')[1][:-4])
    data_idx = frame_idx + index
    with pyexr.open(os.path.join(referencepath, test["referenceName"].format(index=data_idx))) as exr:
        reference = exr.get()[:, :, :3].astype(np.float32)
    reference = torch.from_numpy(reference).permute(2, 0, 1)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    reference = avg_pool(reference)
    return calc_psnr(img, reference)


def main(cfg):
    with open(cfg, 'r') as f:
        config_info = json.load(f)
    model_path = os.path.join(config_info["model"]["path"], config_info["model"]["name"])
    # model = model_nppd.load_from_checkpoint(checkpoint_path=model_path)
    # model = model_kernel_S.load_from_checkpoint(checkpoint_path=model_path, map_location=lambda storage, loc: storage.cuda(0))
    model = model_kernel_S().cuda(0)
    # test_set = baseTestDataset(**config_info)
    # test_set = TestDataset_nppd(**config_info)
    # test_set = TestDataset_ME(**config_info)
    test_set = TestDataset_ME_new(**config_info)

    # output_folder = os.path.join('outputs', test_set.modelInfo["config_name"])
    # ckpt_folder = os.path.join(output_folder, 'ckpt_epoch')
    # ckpt_files = glob.glob(os.path.join(ckpt_folder, "*.ckpt"))
    # def extract_val_loss(filename):
    #     name = os.path.basename(filename)
    #     # Matches 'val_loss=' followed digits, a decimal point, and more digits (e.g., 'val_loss=0.123')
    #     return float(re.search(r'val_loss=(\d+.\d+)', name).group(1))
    # best_model_path = min(ckpt_files, key=lambda x: extract_val_loss(x))

    #t_start = time.time()
    outpath = config_info["test"]["savepath"]
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    first = True
    offset_WL, offset_WR, offset_HT, offset_HB = test_set.augmentdata.padding
    height = test_set.augmentdata.height
    width = test_set.augmentdata.width
    for i, frame in enumerate(loader):
        frame = toTorch(frame, model.device)
        if first:
            first = False if i % 63 == 0 else True
            model.temporal = model.temporal_init(frame)

        with torch.no_grad():
            output = model.test_step(frame)
        output = output[0, :, offset_HT:height+offset_HT, offset_WL:width+offset_WL].cpu()
        # psnr = calcPSNR(output, i, config_info)
        # save_exr_psnr(i, output, outpath, 'predict', str(psnr)[0:5])
        save_exr(i, output, outpath, 'predict')
        # if i == 10:
        #     break


if __name__ == '__main__':
    config_path = "conf/meTest.json"
    main(config_path)