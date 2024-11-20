import torch
import torch.nn.functional as F
import os
import numpy as np
import pyexr
from core.networks.modelt import GrenderModel
from core.networks.modelt import model_kernel_init, model_kernel_L, model_kernel_S, model_kernel_T #, model_ret_L, model_ret_S
import onnx
from thop import profile

def temporal_init(x):
    shape = list(x['color'].shape)
    shape[1] = 38
    return torch.zeros(shape, dtype=x['color'].dtype, device=x['color'].device)


def checkpadding(input_frames):
    _, C, H, W = input_frames['color'].shape
    offset = 0
    if H % 32 != 0:
        offset = (H // 32 + 1) * 32 - H
    padding = (0, 0, offset // 2, offset // 2)
    for key, value in input_frames.items():
        input_frames[key] = F.pad(value, padding, mode='reflect')
    return offset // 2


def savepredictret(img, pad_H, des_path, filename):
    output = img[:, :, pad_H:-pad_H, :]
    os.makedirs(des_path, exist_ok=True)
    image_array = np.transpose(output.cpu().numpy()[0], (1, 2, 0))
    pyexr.write(os.path.join(des_path, filename), image_array)


def transfer2onnx(input_sample, srcpath, despath):
    # model = GrenderModel.load_from_checkpoint(checkpoint_path=srcpath).cuda()
    # model = model_kernel_init().cuda()
    model = model_kernel_S().cuda()

    model.eval()
    with torch.no_grad():
        # flops, params = profile(model, (input_sample['color'], input_sample['depth'], input_sample['normal'], input_sample['albedo'], input_sample['motion'], input_sample['temporal']))
        output1, output2 = model(input_sample['color'], input_sample['depth'], input_sample['normal'], input_sample['albedo'], input_sample['motion'], input_sample['temporal'])
        torch.onnx.export(
            model,
            (input_sample['color'], input_sample['depth'], input_sample['normal'], input_sample['albedo'], input_sample['motion'], input_sample['temporal']),
            despath,
            export_params=True,
            opset_version=17,
            verbose=False,
            input_names=['color', 'depth', 'normal', "albedo", 'motion', 'temporal'],
            output_names=['output1', 'output2'],
            # dynamic_axes={'color': {2: "height", 3: "width"},
            #               'depth': {2: "height", 3: "width"},
            #               'normal': {2: "height", 3: "width"},
            #               'albedo': {2: "height", 3: "width"},
            #               'motion': {2: "height", 3: "width"},
            #               "temporal": {2: "height", 3: "width"},
            #               "output1": {2: "height", 3: "width"},
            #               "output2": {2: "height", 3: "width"}
            #               }
            )
    return output1, output2

def calcFlops(input_sample):
    # model = GrenderModel.load_from_checkpoint(checkpoint_path=srcpath).cuda()
    # model = model_kernel_init().cuda()
    model = model_kernel_S().cuda()

    model.eval()
    with torch.no_grad():
        flops, params = profile(model, (input_sample['color'], input_sample['depth'], input_sample['normal'], input_sample['albedo'], input_sample['motion'], input_sample['temporal']))
        print('flops: ', flops, 'params: ', params)

if __name__ == '__main__':
    input_frames = {
        'color': torch.randn(1, 3, 720, 1280).cuda(),
        'depth': torch.randn(1, 1, 720, 1280).cuda(),
        'normal': torch.randn(1, 3, 720, 1280).cuda(),
        'albedo': torch.randn(1, 3, 720, 1280).cuda(),
        'motion': torch.randn(1, 2, 720, 1280).cuda(),

        'temporal': torch.randn(1, 38, 720, 1280).cuda()
    }
    srcpath = '..\\model\\ckpt\\grender_model_v1.ckpt'
    filepath = ('..\\model\\onnx\\MeModel_Kernel_S_1080.onnx')
    pad = checkpadding(input_frames)
    temporal = temporal_init(input_frames)
    input_frames['temporal'] = temporal

    predict, temporal = transfer2onnx(input_frames, srcpath, filepath)
    path = '..\\test'
    filename = 'predict10090000.exr'
    savepredictret(predict, pad, path, filename)
    model = onnx.load(filepath)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))
    calcFlops(input_frames)
