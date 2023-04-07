import os
import pytorch_lightning as pl
from dataloader.ULane_loader import CUL_datamodule
from model.model import BaseMethod
from utils.checkpointer import Checkpointer

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from torchvision.transforms import Compose,Resize, CenterCrop, ToTensor, Normalize

def main(cfger):
    print(cfger)
    # 1. setup dataset 
    cul_dm = CUL_datamodule(**cfger.dataset, data_ld_args=cfger.dataloader)
    cul_dm.setup()

    # 2. setup model
    #   2.1 while we get model, optimizer or something should already be setup..
    model = BaseMethod(**cfger.model, training_set = cfger.trainer)
    image_trfs = Compose([
        Resize(256),
        CenterCrop(256),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    cul_dm.train_trfs = image_trfs

    mask_trfs = Compose([
        Resize(256),
        CenterCrop(256),
        ToTensor(),
        #Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    cul_dm.train_lab_trfs = mask_trfs

    # 3. setup trainer
    #   3.1 setup callback ( e.g. wandb monitor )
    callbacks = []
    if cfger.mod["use_wandb"]:
        wandb_logger = WandbLogger(**cfger.wandb)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    if cfger.mod["save_checkpoint"]:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            cfger,
            logdir=os.path.join(cfger.ckpt["checkpoint_dir"]),
            frequency=cfger.ckpt["checkpoint_frequency"],
        )
        callbacks.append(ckpt)

    trainer = pl.Trainer(logger=wandb_logger if cfger.mod["use_wandb"] else None ,callbacks=callbacks,
        enable_checkpointing=False,
        devices = cfger.gpu_cfg["devices"],
        accelerator = cfger.gpu_cfg["accelerator"],
        strategy = DDPStrategy(find_unused_parameters=False)
        if cfger.gpu_cfg["strategy"] == "ddp"
        else cfger.gpu_cfg["strategy"],
        accumulate_grad_batches = cfger.gpu_cfg["accumulate_grad_batches"],
        precision=cfger.gpu_cfg["precision"],
        max_epochs = cfger.trainer["epochs"])

    trainer.fit(model, datamodule = cul_dm)


if __name__ == "__main__":
    from easy_configer.Configer import Configer

    cfger = Configer()
    cfger.cfg_from_ini('./config/train.ini')

    main(cfger)
