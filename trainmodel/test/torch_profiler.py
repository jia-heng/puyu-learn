import torch.profiler
from ppq import *
from ppq.api import *
from tqdm import tqdm
import onnx
from onnx import version_converter, helper

def init_input():
    sample_input = [
        {'color': torch.randn(1, 3, 1088, 1920),
         'depth': torch.randn(1, 1, 1088, 1920),
         'normal': torch.randn(1, 3, 1088, 1920),
         'albedo': torch.randn(1, 3, 1088, 1920),
         'motion': torch.randn(1, 2, 1088, 1920),
         'temporal': torch.randn(1, 38, 1088, 1920)
         } for i in range(32)
    ]
    return sample_input

def to_torch(frames):
    for key,value in frames.items():
        frames[key] = value.to('cuda')
    return frames

def quant_onnx_model(sample_input):
    ir = quantize_onnx_model(
        onnx_import_file='..\\model\\onnx\\MeModel_Kernel_T_B13.onnx',
        calib_dataloader=sample_input,
        calib_steps=16,
        do_quantize=False,
        input_shape=None,
        platform=TargetPlatform.PPL_CUDA_FP16,
        collate_fn=lambda x:x.to('cuda'),
        inputs=list(sample_input[0].values())
    )
    return ir


if __name__ == '__main__':
    sample_input = init_input()
    ir = quant_onnx_model(sample_input)
    executor = TorchExecutor(ir)
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name='./performance/'),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            with_stack=True,
    ) as profiler:
        with torch.no_grad():
            for batch_idx in tqdm(range(16), desc='Profiling ...'):
                executor.forward(list(to_torch(sample_input[batch_idx]).values()))
                profiler.step()

    # # 模型路径
    # model_path = '..\\model\\onnx\\MeModel_Kernel_T_W.onnx'
    #
    # # 加载原始模型
    # original_model = onnx.load(model_path)
    # print(f"The model before conversion:\n{original_model}")
    #
    # # 转换模型版本到13
    # converted_model = version_converter.convert_version(original_model, 13)
    # print(f"The model after conversion:\n{converted_model}")
    #
    # # 保存转换后的模型
    # save_model_path = model_path[:-5] + "_opset13.onnx"
    # onnx.save(converted_model, save_model_path)
