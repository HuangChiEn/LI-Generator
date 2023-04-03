import pytorch_lightning as pl
from ULane_loader import CUL_datamodule

def main(cfger):
    # 1. setup dataset 
    cul_dm = CUL_datamodule(cfger.data_dir, cfger.lab_ext, data_ld_args=cfger.dataloader)
    cul_dm.setup()

    # 2. setup model
    #   2.1 while we get model, optimizer or something should already be setup..
    ...

    # 3. setup trainer
    #   3.1 setup callback ( e.g. wandb monitor )
    trainer = pl.Trainer(...) 
    trainer.fit(..., cul_dm)


if __name__ == "__main__":
    from easy_configer.Configer import Configer

    cfger = Configer()
    cfger.cfg_from_ini('./config/train.ini')

    main(cfger)
