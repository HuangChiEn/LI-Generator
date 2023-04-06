import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from generator import Generator
from utils.LRscheduler import LinearWarmupCosineAnnealingLR

class BaseMethod(pl.LightningModule):
    def __init__(self,
                 clip_version,
                 mask_num_blocks,
                 mask_out_dim,
                 norm_name,
                 G_num_blocks,
                 G_input_shape,
                 G_input_dim,
                 no_masks,
                 num_mask_channels
                 ):
        super().__init__()
        self.encoder = Encoder(clip_version, mask_num_blocks, num_mask_channels , mask_out_dim, norm_name)

        self.mix_layer = nn.Linear(512 + mask_out_dim, input_dim)

        self.generator = Generator(G_num_blocks, norm_name, G_input_shape, G_input_dim, no_masks, num_mask_channels)

        self.mask_loss_fun = nn.CrossEntropyLoss()
    @property
    def learnable_params(self):
        return self.encoder.learnable_params + self.generator.learnable_params + [{"name": "mix_block", "params": self.mix_layer.parameters()}]
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.learnable_params,
            lr=self.lr,
            betas=(
                0.9,
                0.999
            ),
            eps=1e-8,
            weight_decay=0.2
        )

        lr_scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_epochs,
                warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
            ),
            "interval": self.scheduler_interval,
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def forward(self, X, Y):
        x = self.encoder(X, Y)
        x = self.mix_layer(x)
        x, mask = self.generator(x)

        return x, mask

    def training_step(self, batch, batch_idx: int):
        img, mask = batch
        pre_img, pre_mask = self(img, mask)

        mask_loss = self.mask_loss_fun(pre_mask, mask)
        img_loss = torch.nn.MSELoss(pre_img, img)

        self.log('train_mask_loss', mask_loss)
        self.log('train_img_loss', img_loss)
        return mask_loss + img_loss

    def validation_step(self, val_batch, batch_idx):
        img, mask = batch
        pre_img, pre_mask = self(img, mask)

        mask_loss = self.mask_loss_fun(pre_mask, mask)
        img_loss = torch.nn.MSELoss(pre_img, img)

        self.log('val_mask_loss', mask_loss)
        self.log('val_img_loss', img_loss)





