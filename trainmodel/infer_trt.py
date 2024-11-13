import os
import torch
import numpy as np
import pyexr
import tensorrt as trt
from core import common
from utils.trt_utils import get_read_permission
from dataLoader import TestDataset
import json

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

    def set_inputdata(self, frames):
        input_names = {'color': 3, 'depth': 1, 'normal': 3, "diffuse": 3, 'motion': 2, 'temporal': 38}
        for i, (key, value) in enumerate(frames.items()):
            input_dims = value.shape
            self.context.set_input_shape(key, input_dims)
            # if C == 1:
            #     item_reshaped = item.reshape(H*W)
            # else:
            #     item_reshaped = item.reshape(C, H * W)
            self.inputs[i].host = value.ravel()

    def allocate_buffers(self):
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine, 0)
        self.context = self.engine.create_execution_context()

    def free_buffers(self):
        common.free_buffers(self.inputs, self.outputs, self.stream)

    def infer(self, input_data):

        self.set_inputdata(input_data)
        output = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        self.temporal = output[1].reshape((1, 38, 736, 1280))
        predict = output[0].reshape((1, 3, 736, 1280))
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

def save(idx, image, path, imgType):
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


if __name__ == '__main__':
    model_path = '.\\model\\trt\\grender_model_fp16_0929_1k.engine'
    inferer = InferImageME(model_path, TRT_LOGGER)
    config_path = "conf/grenderTest.json"
    with open(config_path, 'r') as f:
        config_info = json.load(f)
    test_set = TestDataset(**config_info)
    outpath = config_info["test"]["savepath"]
    # loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=True,
    # )
    length = len(test_set)
    inferer.allocate_buffers()
    for i in range(10):
        frame = test_set[i]
        output = inferer(frame)
        save(i, output, outpath, 'predict_fp16_1k')
    inferer.free_buffers()