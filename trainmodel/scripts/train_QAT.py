import torch
import torch.utils.data
from core.networks.modelt import model_kernel_F
from core.loader.TestDataLoader import TestDataset_nppd
import json
import time
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

def apply_pad(frames, padding=(0,0,0,64)):
    """ C H W """
    for key, value in frames.items():
        frames[key] = F.pad(value.unsqueeze(0), padding, mode='reflect').squeeze(0)
        # frames[key] = F.pad(value.unsqueeze(0), self.padding, mode='constant', value=1).squeeze(0)
    return frames

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

def temporal_init(frame):
    shape = list(frame['reference'].shape)
    shape[1] = 19
    return torch.zeros(shape, dtype=frame['reference'].dtype, device=frame['reference'].device)

def quantize_model(model, quant_cfg, dataloader_re=None):
    def forward_loop(model):
        first = True
        for i, data in enumerate(tqdm(dataloader_re)):
            # data = apply_pad(data)
            frame = toTorch(data, model.device)
            if first:
                first = False if i % 63 == 0 else True
                temporal = temporal_init(frame)
            predict, temporal = model(frame['color'], frame['depth'], frame['normal'], frame['albedo'], frame['motion'],
                                      temporal)
            temporal = temporal.detach()

    print("Starting quantization...")
    start_time = time.time()
    model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    end_time = time.time()
    print(f"Quantization done. Total time used: {end_time - start_time}s")
    return model

def transfer_quant_onnx():
    cfg = "..\\conf\\retrain.json"
    with open(cfg, 'r') as f:
        config_info = json.load(f)
    test_set = TestDataset_nppd(**config_info)
    dataloader_re = DataLoader(dataset=test_set,
                               batch_size=1,  # 批量大小
                               shuffle=False,  # 是否打乱数据
                               num_workers=4)  # 使用多少进程来加载数据
    dataloader_re.dataset.augmentdata.padding = (0,0,0,64)
    config = mtq.INT8_DEFAULT_CFG
    for i in range(5):  # 假设有5个encoder层
        for quantizer_type in ["input_quantizer", "output_quantizer", "weight_quantizer"]:
            config["quant_cfg"]["encoder." + str(i) + "." + quantizer_type] = {"enable": False}

    srcpath = '../model/ckpt/epoch=111-val_loss=0.327464.ckpt'
    filepath = '../model/onnx/MeModel_Kernel_F1_splat.onnx'
    model = model_kernel_F.load_from_checkpoint(checkpoint_path=srcpath, map_location=lambda storage, loc: storage.cuda(0))
    quant_model = quantize_model(model, config, dataloader_re)
    mtq.print_quant_summary(quant_model)
    # Save the modelopt quantizer states
    torch.save(mto.modelopt_state(quant_model), "../model/ckpt/modelopt_quantizer_states.pt")
    # train model

    quant_model.eval()
    for n in dataloader_re:
        frame = n
        break
    input_sample = toTorch(frame, model.device)
    temporal = temporal_init(input_sample)
    # torch.utils.cpp_extension.load()
    with torch.no_grad():
        # flops, params = profile(model, (input_sample['color'], input_sample['depth'], input_sample['normal'], input_sample['albedo'], input_sample['motion'], input_sample['temporal']))
        output1, output2 = quant_model(input_sample['color'], input_sample['depth'], input_sample['normal'], input_sample['albedo'], input_sample['motion'], temporal)
        torch.onnx.export(
            quant_model,
            (input_sample['color'], input_sample['depth'], input_sample['normal'], input_sample['albedo'], input_sample['motion'], temporal),
            filepath,
            export_params=True,
            opset_version=17,
            verbose=False,
            input_names=['color', 'depth', 'normal', "albedo", 'motion', 'temporal'],
            output_names=['output1', 'output2'],
            do_constant_folding=True,
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
    return True

if __name__ == '__main__':
    # train_QAT()
    transfer_quant_onnx()