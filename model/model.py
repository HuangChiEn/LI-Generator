import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from generator import Generator

class BaseMethod(pl.LightningModule):
    def __init__(self,
                 clip_version,
                 mask_num_blocks,
                 mask_out_dim,
                 norm_name,
                 G_num_blocks,
                 input_shape,
                 input_dim,
                 no_masks,
                 num_mask_channels
                 ):
        super().__init__()
        self.encoder = Encoder(clip_version, mask_num_blocks, num_mask_channels , mask_out_dim, norm_name)
        self.generator = Generator(G_num_blocks, norm_name, input_shape, input_dim, no_masks, num_mask_channels)

        self.mix_layer = nn.Linear(512 + mask_out_dim, input_dim)

        self.mask_loss_fun = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        pass

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

        return mask_loss + img_loss







