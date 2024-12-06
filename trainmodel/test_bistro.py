import numpy as np
from core.networks.modelnet import BaseModel
from core.networks.modelMe import model_kernel_nppd, model_kernel_nppd_mean, model_kernel_T_B_nppd
import torch
import os
import glob
import re
import pyexr
from core.loader.TestDataLoader import baseTestDataset, TestDataset_nppd, TestDataset_ME, TestDataset_ME_new
from core.loader.dataloader_torch import TestSampleDataset
import json
from core.loader.misc import ACES
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
    output_path = path
    os.makedirs(output_path, exist_ok=True)
    filename = '{imgType}_{idx:04d}.png'.format(imgType=imgType, idx=idx)
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
    output_path = path
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

def main(cfg):
    with open(cfg, 'r') as f:
        config_info = json.load(f)
    model_path = os.path.join(config_info["model"]["path"], config_info["model"]["name"])
    model = model_kernel_T_B_nppd.load_from_checkpoint(checkpoint_path=model_path, map_location=lambda storage, loc: storage.cuda(0))
    test_set = TestSampleDataset(**config_info)

    outpath = config_info["test"]["savepath"]
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    first = True
    for i, frame in enumerate(loader):
        frame = toTorch(frame, model.device)
        if first:
            first = False if i % 63 == 0 else True
            model.temporal = model.temporal_init(frame)

        with torch.no_grad():
            output = model.test_step(frame)
        output = output[0].cpu()
        psnr = calc_psnr(output, frame['reference'][0].cpu())
        # save_exr_psnr(i, output, outpath, 'predict', str(psnr)[0:5])
        save_png(i, output, outpath, 'predict')

if __name__ == '__main__':
    config_path = "conf/nppdTest.json"
    main(config_path)