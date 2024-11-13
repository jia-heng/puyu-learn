import os
import tensorrt as trt
import argparse
from tensorrt import Runtime
from utils.trt_utils import TensorRTEngineBuilder, save_engine
# import json
import onnx

def parse_option():
    model_path = '..\\model\\onnx\\grender_model_v1_1009_temp.onnx'
    output_path = '..\\model\\trt\\grender_model_fp16_1009_temp.engine'

    parser = argparse.ArgumentParser('TensorRT engine build script for cnn and Transformer', add_help=False)
    parser.add_argument("--model_path", '-m', default=model_path, help='Onnx model file')
    parser.add_argument("--output_path", '-o', default=output_path, help='TensorRT engine output path')
    parser.add_argument("--tcf", default='./trt_timing_cache',
                        help="Path to tensorrt build timeing cache file, only available for tensorrt 8.0 and later", required=False)
    parser.add_argument('--verbose', '-v',  default='-v', action='store_true', help='enable verbose output (for debugging)')
    parser.add_argument('--mode', choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument("--opt_size", nargs=2, type=int, default=(736, 1280))
    parser.add_argument("--min_size", nargs=2, type=int, default=(736, 1280))
    parser.add_argument("--max_size", nargs=2, type=int, default=(736, 1280))
    args = parser.parse_args()
    return args


def build_trt_engine():
    args = parse_option()
    model = onnx.load(args.model_path)
    onnx.checker.check_model(model)
    logger = trt.Logger(trt.Logger.VERBOSE)
    runtime: Runtime = trt.Runtime(logger)
    trt_builder = TensorRTEngineBuilder(
        onnx_file_path=args.model_path,
        logger=logger,
        workspace_size=2 << 30,
        mode=args.mode,
    )
    profile = trt_builder.profile
    input_names = {'color': 3, 'depth': 1, 'normal': 3, "diffuse": 3, 'motion': 2, 'temporal': 38}
    for key, value in input_names.items():
        # 对于固定尺寸，最小、最优和最大形状都设置为固定形状
        profile.set_shape(key,
                          (1, value, args.min_size[0], args.min_size[1]),
                          (1, value, args.opt_size[0], args.opt_size[1]),
                          (1, value, args.max_size[0], args.max_size[1]),
                          )
    # output_names = {'output1': 3, 'output2': 38}
    # for key, value in output_names.items():
    #     # 对于固定尺寸，最小、最优和最大形状都设置为固定形状
    #     profile.set_shape(key,
    #                       (1, value, args.min_size[0], args.min_size[1]),
    #                       (1, value, args.opt_size[0], args.opt_size[1]),
    #                       (1, value, args.max_size[0], args.max_size[1]),
    #                       )
    engine = trt_builder.create_engine(
        runtime=runtime,
        timing_cache_file=args.tcf)
    save_engine(engine, args.output_path)


def checkOnnxFile(path):
    if not os.path.exists(path):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")


if __name__ == '__main__':
    build_trt_engine()
