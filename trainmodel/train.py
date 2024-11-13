import os
from core.lightning import MidepochCheckpoint
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import datetime
import json
from core.loader.TrainDataLoader import TrainingSampleLoader_L
from core.networks.modelnet import BaseModel, model_ME, model_nppd
import glob
import re

def load_pretrained_weights(model, checkpoint_path):
    # 加载预训练模型的状态字典
    pretrained = BaseModel.load_from_checkpoint(checkpoint_path)
    pretrained_dict = pretrained.state_dict()

    # 过滤掉不需要的权重
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

    # 更新现有的模型状态字典
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def main(cfg):
    # current_directory = os.getcwd()
    # parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    # os.chdir(parent_directory)
    with open(cfg, 'r') as f:
        config_info = json.load(f)
    output_folder = os.path.join('outputs', config_info["model"]["config_name"])
    os.makedirs(output_folder, exist_ok=True)
    dm = TrainingSampleLoader_L(config_info)

    # ckpt_folder = os.path.join(output_folder, 'ckpt_epoch')
    # ckpt_files = glob.glob(os.path.join(ckpt_folder, "*.ckpt"))
    # def extract_val_loss(filename):
    #     name = os.path.basename(filename)
    #     # Matches 'val_loss=' followed digits, a decimal point, and more digits (e.g., 'val_loss=0.123')
    #     return float(re.search(r'val_loss=(\d+.\d+)', name).group(1))
    # best_model_path = min(ckpt_files, key=lambda x: extract_val_loss(x))
    # modelsrc = BaseModel.load_from_checkpoint(checkpoint_path=best_model_path)
    model_path = os.path.join(config_info["model2"]["path"], config_info["model2"]["name"])
    model = model_ME()
    # model = BaseModel()
    # model = load_pretrained_weights(model, best_model_path)

    checkpoint_time = MidepochCheckpoint(
        dirpath=os.path.join(output_folder, 'ckpt_resume'),
        train_time_interval=datetime.timedelta(minutes=10),
        filename='last',
        enable_version_counter=False,
        save_on_train_epoch_end=True
    )
    # Should work but ModelCheckpoint has been super buggy for me
    # Somehow saves more than 10 checkpoints, and sometimes just stops saving altogether
    # checkpoint_epoch = ModelCheckpoint(
    #     dirpath=os.path.join(output_folder, 'ckpt_epoch'),
    #     every_n_epochs=1,
    #     save_top_k=10,
    #     monitor='val_loss',
    #     mode='min',
    #     filename='{epoch:02d}-{val_loss:.6f}',
    # )

    # Save every epoch instead
    # Needs manual cleanup of old checkpoints but at least works
    checkpoint_epoch = ModelCheckpoint(
        dirpath=os.path.join(output_folder, 'ckpt_epoch'),
        every_n_epochs=1,
        save_top_k=-1,
        filename='{epoch:02d}-{val_loss:.6f}',
    )
    logger = TensorBoardLogger(
        save_dir=output_folder,
        name='logs',
        version='',  # save to output_folder/logs directly
    )
    trainer = L.Trainer(
        max_epochs=400,
        precision='16-mixed',
        callbacks=[checkpoint_time, checkpoint_epoch],
        logger=logger,
        use_distributed_sampler = False,
        # devices=1
    )
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path='last'  # looks up last.ckpt from checkpoint_time callback
    )


if __name__ == '__main__':
    # config_path = "conf/grenderTrain.json"
    config_path = "conf/meTrain.json"
    main(config_path)


# # 初始化模型
# model = BaseModel()
#
# # 冻结encoder中的所有参数
# for param in model.encoder.parameters():
#     param.requires_grad = False
#
# # 冻结PartitioningPyramid中的所有参数
# for param in model.filter.parameters():
#     param.requires_grad = False
#
# # 冻结ConvUNet中的所有参数
# for param in model.weight_predictor.parameters():
#     param.requires_grad = False
#
# # 现在，只有未被冻结的参数会在训练过程中更新