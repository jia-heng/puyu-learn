import torch
import torch.nn.functional as F
from core.networks.modelt import GrenderModel
import os
import numpy as np
import pyexr
import tensorrt as trt
from core import common
from utils.trt_utils import get_read_permission
from core.loader.TestDataLoader import TestDataset_trt
from core.loader.dataloader_torch import TestDataset_torch
import json
import time


class TorchInfer:
    def __init__(self, model_path):
        self.model = GrenderModel.load_from_checkpoint(checkpoint_path=model_path).cuda().eval()
        self.temporal = None

    def temporal_init(self, x):
        shape = list(x['color'].shape)
        shape[1] = 38
        return torch.zeros(shape, dtype=x['color'].dtype, device=x['color'].device)

    def checkpadding(self, input_frames):
        B, C, H, W = input_frames['color'].shape
        offset = 0
        if H % 32 != 0:
            offset = (H // 32 + 1) * 32 - H
        padding = (0, 0, offset // 2, offset - offset // 2)
        for key, value in input_frames.items():
            input_frames[key] = F.pad(value, padding, mode='reflect')
        return offset // 2


    def infer(self, input_data):
        T1 = time.perf_counter()
        output = self.model(input_data['color'], input_data['depth'], input_data['normal'], input_data['diffuse'], input_data['motion'], self.temporal)
        T2 = time.perf_counter()
        predict = output[0]
        self.temporal = output[1].detach()
        print("infer in {:.2f} ms.".format((T2 - T1) * 1000))
        # common.free_buffers(inputs, outputs, stream)
        return predict

    def __call__(self, input_data):
        pad = self.checkpadding(input_data)
        if self.temporal is None:
            self.temporal = self.temporal_init(input_data)
        predict = self.infer(input_data)
        ret = predict[:, :, pad:-pad, :]
        return ret


import onnxruntime as ort
class OnnxInfer:
    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        self.ort_sess = ort.InferenceSession(model_path, opts, providers=["CUDAExecutionProvider"])
        print(self.ort_sess.get_providers())
        self.temporal = None

    def temporal_init(self, x):
        shape = list(x['color'].shape)
        shape[1] = 38
        return torch.zeros(shape, dtype=x['color'].dtype, device=x['color'].device)

    def checkpadding(self, input_frames):
        B, C, H, W = input_frames['color'].shape
        offset = 0
        if H % 32 != 0:
            offset = (H // 32 + 1) * 32 - H
        padding = (0, 0, offset // 2, offset - offset // 2)
        for key, value in input_frames.items():
            input_frames[key] = F.pad(value, padding, mode='reflect')
        return offset // 2


    def infer(self, input_data):
        # onnx_input = {
        #     'l_color_': input_data["color"],
        #     'l_depth_': input_data["depth"],
        #     'l_normal_': input_data["normal"],
        #     'l_diffuse_': input_data["diffuse"],
        #     'l_motion_': input_data["motion"],
        #     'l_temporal_': input_data["temporal"]
        # }
        # onnx_input = {
        #     'l_color_': ort.OrtValue.ortvalue_from_numpy(input_data["color"], 'cuda', 0),
        #     'l_depth_': ort.OrtValue.ortvalue_from_numpy(input_data["depth"], 'cuda', 0),
        #     'l_normal_': ort.OrtValue.ortvalue_from_numpy(input_data["normal"], 'cuda', 0),
        #     'l_diffuse_': ort.OrtValue.ortvalue_from_numpy(input_data["diffuse"], 'cuda', 0),
        #     'l_motion_': ort.OrtValue.ortvalue_from_numpy(input_data["motion"], 'cuda', 0),
        #     'l_temporal_': ort.OrtValue.ortvalue_from_numpy(input_data["temporal"], 'cuda', 0)
        # }
        # onnx_input = {
        #     'l_color_': input_data["color"].contiguous(),
        #     'l_depth_': input_data["depth"].contiguous(),
        #     'l_normal_': input_data["normal"].contiguous(),
        #     'l_diffuse_': input_data["diffuse"].contiguous(),
        #     'l_motion_': input_data["motion"].contiguous(),
        #     'l_temporal_': input_data["temporal"].contiguous()
        # }
        io_binding = self.ort_sess.io_binding()
        for key, value in input_data.items():
            io_binding.bind_input(key, device_type='cuda', device_id=0, element_type=np.float32, shape=tuple(value.shape), buffer_ptr=value.data_ptr())
        output1 = torch.empty([1, 3, 736, 1280], dtype=torch.float32, device='cuda:0').contiguous()
        output2 = torch.empty([1, 38, 736, 1280], dtype=torch.float32, device='cuda:0').contiguous()
        io_binding.bind_output('output1', device_type='cuda', device_id=0, element_type=np.float32, shape=(1, 3, 736, 1280), buffer_ptr=output1.data_ptr())
        io_binding.bind_output('output2', device_type='cuda', device_id=0, element_type=np.float32, shape=(1, 38, 736, 1280), buffer_ptr=output2.data_ptr())
        T1 = time.perf_counter()
        self.ort_sess.run_with_iobinding(io_binding)
        # output = self.ort_sess.run(None, onnx_input)
        T2 = time.perf_counter()
        output = io_binding.copy_outputs_to_cpu()
        predict = output[0]
        self.temporal = torch.from_numpy(output[1]).cuda()
        print("infer in {:.2f} ms.".format((T2 - T1) * 1000))
        # common.free_buffers(inputs, outputs, stream)
        return predict

    def __call__(self, input_data):
        pad = self.checkpadding(input_data)
        if self.temporal is None:
            self.temporal = self.temporal_init(input_data)
        input_data['temporal'] = self.temporal
        predict = self.infer(input_data)
        ret = predict[:, :, pad:-pad, :]
        return ret


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
class InferImageME:
    def __init__(self, model_path, logger=TRT_LOGGER):
        self.logger = logger
        if not os.path.isfile(model_path):
            self.logger.critical("{} is not a regular file".format(model_path))
            raise RuntimeError("{} is not a regular file".format(model_path))
        self.engine = self.load_engine(model_path)
        if self.engine is None:
            self.logger.critical("Load model failed.")
            raise RuntimeError("Load model failed.")
        self.temporal = None

    def load_engine(self, filename):
        flag, mode = get_read_permission()
        print('TRT model path: ', filename)
        f = os.fdopen(os.open(filename, flag, mode), 'rb')
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def temporal_init(self, x):
        shape = list(x['color'].shape)
        shape[1] = 38
        return np.zeros(shape, dtype=x['color'].dtype)

    def checkpadding(self, input_frames):
        B, C, H, W = input_frames['color'].shape
        offset = 0
        if H % 32 != 0:
            offset = (H // 32 + 1) * 32 - H
        padding = ((0, 0), (0, 0), (offset // 2, offset - offset // 2), (0, 0))
        for key, value in input_frames.items():
            input_frames[key] = np.pad(value, padding, mode='reflect')
        return offset // 2

    def set_inputdata(self, context, frames, inputs):
        input_names = {'color': 3, 'depth': 1, 'normal': 3, "diffuse": 3, 'motion': 2, 'temporal': 38}
        for i, (key, value) in enumerate(frames.items()):
            input_dims = value.shape
            context.set_input_shape(key, input_dims)
            # if C == 1:
            #     item_reshaped = item.reshape(H*W)
            # else:
            #     item_reshaped = item.reshape(C, H * W)
            inputs[i].host = value.ravel()

    def infer(self, input_data):
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine, 0)
        context = self.engine.create_execution_context()
        # T1 = time.perf_counter()
        self.set_inputdata(context, input_data, inputs)
        output = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        self.temporal = output[1].reshape((1, 38, 736, 1280))
        predict = output[0].reshape((1, 3, 736, 1280))
        # T2 = time.perf_counter()
        # print("infer in {:.2f} ms.".format((T2 - T1) * 1000))
        # common.free_buffers(inputs, outputs, stream)
        return predict

    def __call__(self, input_data):
        pad = self.checkpadding(input_data)
        if self.temporal is None:
            self.temporal = self.temporal_init(input_data)
        input_data['temporal'] = self.temporal
        predict = self.infer(input_data)
        ret = predict[:, :, pad:-pad, :]
        return ret


def savepredictret(img, des_path, filename):
    os.makedirs(des_path, exist_ok=True)
    image_array = np.transpose(img[0], (1, 2, 0))
    pyexr.write(os.path.join(des_path, filename), image_array)

def save_tensor(idx, image, path, imgType):
    # image = np.transpose(image.cpu().numpy()[0], (1,2,0))
    # image = (ACES(image)*255).astype(np.uint8)
    # self.save_pool.apply_async(iio.imwrite, [file, image])
    imgType = str(imgType)
    output_path = os.path.join(path, 'v2')
    os.makedirs(output_path, exist_ok=True)
    filename = '{imgType}_{idx:04d}.exr'.format(imgType=imgType, idx=idx)
    file_path = os.path.join(output_path, filename)
    image_array = np.transpose(image.detach().cpu().numpy()[0], (1, 2, 0))
    pyexr.write(file_path, image_array)

def save_numpy(idx, image, path, imgType):
    # image = np.transpose(image.cpu().numpy()[0], (1,2,0))
    # image = (ACES(image)*255).astype(np.uint8)
    # self.save_pool.apply_async(iio.imwrite, [file, image])
    imgType = str(imgType)
    output_path = os.path.join(path, 'v2')
    os.makedirs(output_path, exist_ok=True)
    filename = '{imgType}_{idx:04d}.exr'.format(imgType=imgType, idx=idx)
    file_path = os.path.join(output_path, filename)
    image_array = np.transpose(image[0], (1, 2, 0))
    pyexr.write(file_path, image_array)

def torch_infer_test(model_path):
    config_path = "conf/meTest.json"
    with open(config_path, 'r') as f:
        config_info = json.load(f)
    test_set = TestDataset_torch(**config_info)
    outpath = config_info["test"]["savepath"]
    model_path = config_info["model"]["path"] + "\\" + config_info["model"]["name"]
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    torchinfer = TorchInfer(model_path)
    length = len(test_set)
    i = 0
    with torch.no_grad():
        for frame in loader:  # 注意enumerate的起始索引和解包
            for key, value in frame.items():
                frame[key] = value.cuda()
            # from torch.profiler import profile, record_function, ProfilerActivity
            # with profile(activities=[
            #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with record_function("model_inference"):
            output = torchinfer(frame)  # 执行推理
            save_tensor(i, output, outpath, 'torchinfer1031_1k')  # 保存结果
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            i+=1
            if i == 10:
                break

def onnx_infer_test(model_path):
    config_path = "conf/grenderTest.json"
    with open(config_path, 'r') as f:
        config_info = json.load(f)
    test_set = TestDataset_torch(**config_info)
    outpath = config_info["test"]["savepath"]
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    onnxinfer = OnnxInfer(model_path)
    i = 0
    with torch.no_grad():
        for frame in loader:  # 注意enumerate的起始索引和解包
            for key, value in frame.items():
                frame[key] = value.cuda()
            output = onnxinfer(frame)
            save_numpy(i, output, outpath, 'onnx_1011_1k')  # 保存结果
            i+=1
            if i == 3:
                break

def trt_infer_test(model_path):
    inferer = InferImageME(model_path, TRT_LOGGER)
    config_path = "conf/grenderTest.json"
    with open(config_path, 'r') as f:
        config_info = json.load(f)
    test_set = TestDataset_trt(**config_info)
    outpath = config_info["test"]["savepath"]
    # loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=True,
    # )
    length = len(test_set)
    for i in range(1):
        frame = test_set[i]
        output = inferer(frame)
        save_numpy(i, output, outpath, 'predict_fp16_1k')


if __name__ == '__main__':
    model_torch_path = '.\\model\\ckpt\\grender_model_v1.ckpt'
    model_onnx_path = '.\\model\\onnx\\grender_model_v1_1011.onnx'
    model_engine_path = '.\\model\\trt\\grender_model_fp16_0929_1k.engine'
    filepath = '.\\model\\onnx\\grender_model_v1_1008.onnx'

    torch_infer_test(model_torch_path)
    # onnx_infer_test(model_onnx_path)
    # trt_infer_test(model_engine_path)
