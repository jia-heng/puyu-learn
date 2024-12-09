import torch.profiler
from ppq import *
from ppq.api import *
from ppq.core import PPQ_CONFIG
from tqdm import tqdm
import onnx
from core.networks.modelt import model_kernel_init, model_kernel_L, model_kernel_S, model_kernel_T, model_kernel_T_B_mmcv, model_kernel_T_B
import json


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
        onnx_import_file='..\\model\\onnx\\MeModel_Kernel_T_B_mmcv.onnx',
        calib_dataloader=sample_input,
        calib_steps=16,
        do_quantize=False,
        input_shape=None,
        platform=TargetPlatform.PPL_CUDA_FP16,
        collate_fn=lambda x:x.to('cuda'),
        inputs=list(sample_input[0].values())
    )
    return ir


def analyse(cfg):
    PPQ_CONFIG.USING_CUDA_KERNEL = False
    PPQ_CONFIG.EXPORT_PPQ_INTERNAL_INFO = True
    PPQ_CONFIG.PPQ_DEBUG = True
    # ------------------------------------------------------------
    # Step - 1. 加载校准集与模型
    # ------------------------------------------------------------
    BATCHSIZE = 32
    INPUT_SHAPE = None
    DEVICE = 'cuda'
    PLATFORM = TargetPlatform.TRT_INT8
    CALIBRATION = init_input()
    QS = QuantizationSettingFactory.default_setting()

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return list(x.to(DEVICE) for x in batch.values())

    with open(cfg, 'r') as f:
        config_info = json.load(f)

    # model_path = os.path.join(config_info["model"]["path"], config_info["model"]["name"])
    # model = model_kernel_T_B_mmcv.load_from_checkpoint(checkpoint_path=model_path,
    #                                                    map_location=lambda storage, loc: storage.cuda(0))
    # model = model.to(DEVICE)

    # ------------------------------------------------------------
    # Step - 2. 执行首次量化，完成逐层误差分析
    # ------------------------------------------------------------
    # 如果有激活函数，需要一起添加非量化表中
    QS.dispatching_table.append('/encoder/encoder.4/Conv', TargetPlatform.FP16)
    quantized = quantize_onnx_model(
        onnx_import_file='..\\model\\onnx\\MeModel_Kernel_T_B_mmcv.onnx',
        calib_dataloader=CALIBRATION,
        calib_steps=32,
        input_shape=None,
        platform=PLATFORM,
        collate_fn=collate_fn,
        setting = QS,
        inputs=list(CALIBRATION[0].values())
    )
    # quantized = quantize_torch_model(
    #     model=model,
    #     calib_dataloader=CALIBRATION,
    #     calib_steps=32,
    #     input_shape=None,
    #     collate_fn=collate_fn,
    #     platform=PLATFORM,
    #     setting=QS,
    #     inputs=list(sample_input[0].values()),
    #     onnx_export_file='Output/onnx.model',
    #     device=DEVICE,
    #     verbose=0
    # )

    reports = layerwise_error_analyse(
        graph=quantized, running_device=DEVICE,
        collate_fn=collate_fn, dataloader=CALIBRATION)

    reports = graphwise_error_analyse(
        graph=quantized, running_device=DEVICE,
        collate_fn=collate_fn, dataloader=CALIBRATION)
    ONNX_OUTPUT_PATH = '..\\model\\onnx\\MeModel_Kernel_T_B_mmcv_quant.onnx'
    export_ppq_graph(
        graph=quantized, platform=TargetPlatform.ONNXRUNTIME,
        graph_save_to=ONNX_OUTPUT_PATH)

if __name__ == '__main__':
    cfg = "../conf/meTest.json"
    analyse(cfg)
