import numpy as np
from core.networks.modelnet import BaseModel, model_nppd, model_ME
import torch
import os
import glob
import re
import pyexr
from core.loader.TestDataLoader import baseTestDataset, TestDataset_nppd, TestDataset_ME
import json
from core.loader.misc import ACES
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

    image = np.transpose(image.cpu().numpy()[0], (1,2,0))
    image = (ACES(image)*255).astype(np.uint8)
    iio.imwrite(file_path, image)

def save_exr(idx, image, path, imgType):
    # image = np.transpose(image.cpu().numpy()[0], (1,2,0))
    # image = (ACES(image)*255).astype(np.uint8)
    # self.save_pool.apply_async(iio.imwrite, [file, image])
    imgType = str(imgType)
    output_path = os.path.join(path, 'exr')
    os.makedirs(output_path, exist_ok=True)
    filename = '{imgType}_{idx:04d}.exr'.format(imgType=imgType, idx=idx)
    file_path = os.path.join(output_path, filename)
    image_array = np.transpose(image.cpu().numpy()[0], (1, 2, 0))
    pyexr.write(file_path, image_array)
    # cv2.imwrite(r'C:\Users\j00589466\blender_results\bmfr_modern_living_rooom5\f_warp3.exr', f_warp)
    #pyexr.write(os.path.join(output_path, '{imgType}_{idx}.exr'.format(imgType, idx)), image, compression=pyexr.PXR24_COMPRESSION)

def main(cfg):
    # current_directory = os.getcwd()
    # parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    # os.chdir(parent_directory)
    with open(cfg, 'r') as f:
        config_info = json.load(f)
    model_path = os.path.join(config_info["model"]["path"], config_info["model"]["name"])
    # model = model_nppd.load_from_checkpoint(checkpoint_path=model_path)
    model = model_ME.load_from_checkpoint(checkpoint_path=model_path)

    # test_set = baseTestDataset(**config_info)
    # test_set = TestDataset_nppd(**config_info)
    test_set = TestDataset_ME(**config_info)

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
        save_exr(i, output[:, :, offset_HT:height+offset_HT, offset_WL:width+offset_WL], outpath, 'predict')
        # if i == 10:
        #     break


if __name__ == '__main__':
    config_path = "conf/meTest.json"
    main(config_path)