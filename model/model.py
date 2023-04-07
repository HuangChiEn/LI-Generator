import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .generator import Generator
from utils.LRscheduler import LinearWarmupCosineAnnealingLR
import numpy as np
#from torchmetrics.image.fid import FrechetInceptionDistance
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
                 num_mask_channels,
                 training_set
                 ):
        super().__init__()
        self.encoder = Encoder(clip_version, mask_num_blocks, num_mask_channels , mask_out_dim, norm_name)

        self.mix_layer = nn.Linear(512 + mask_out_dim, G_input_dim)

        self.generator = Generator(G_num_blocks, norm_name, G_input_shape, G_input_dim, no_masks, num_mask_channels)

        self.mask_loss_fun = nn.CrossEntropyLoss()
        self.reconstruction_loss = torch.nn.MSELoss()
        #self.fid = FrechetInceptionDistance(feature=64)
        self.training_set = training_set

        self.validation_step_outputs = []

    @property
    def learnable_params(self):
        return self.encoder.learnable_params + self.generator.learnable_params + [{"name": "mix_block", "params": self.mix_layer.parameters()}]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{"params":self.encoder.mask_encoder.parameters() }, {"params":self.generator.parameters()}, {"params":self.mix_layer.parameters()}],
            lr=self.training_set["lr"],
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
                warmup_epochs=self.training_set["warmup_epochs"],
                max_epochs=self.training_set["epochs"],
                warmup_start_lr=0 if self.training_set["warmup_epochs"] > 0 else self.training_set["lr"],
            )
        }

        return [optimizer], [lr_scheduler]

    def forward(self, X, Y):
        X = X.clone().resize_(X.shape[0],X.shape[1], 224, 224)
        x, middle_feats = self.encoder(X, Y)
        x = self.mix_layer(x)
        x, mask = self.generator(x.unsqueeze(-1).unsqueeze(-1), middle_feats)
        return x, mask

    def training_step(self, batch, batch_idx: int):
        img, mask = batch
        #mask = torch.nn.functional.one_hot(mask, 2)

        pre_img, pre_mask = self(img, mask)

        #print(pre_img.shape, img.shape)
        mask_loss = self.mask_loss_fun(pre_mask , mask)
        img_loss = self.reconstruction_loss(pre_img, img)

        self.log('train_mask_loss', mask_loss)
        self.log('train_img_loss', img_loss)
        return mask_loss + img_loss
    #
    def validation_step(self, val_batch, batch_idx):
        img, mask = val_batch
        pre_img, pre_mask = self(img, mask)
        self.validation_step_outputs.append({"img":img[0], "mask":mask[0],"pre_img":pre_img[0], "pre_mask":pre_mask[0]})
        mask_loss = self.mask_loss_fun(pre_mask, mask)
        img_loss = self.reconstruction_loss(pre_img, img)

        # self.fid.update(img, real=True)
        # self.fid.update(pre_img, real=False)
        # fid_score = self.fid.compute()

        self.log('val_mask_loss', mask_loss)
        self.log('val_img_loss', img_loss)
        #self.log('val_fid_score', fid_score)

    def on_validation_epoch_end(self):
        # do something with all preds
        ...
        out = self.validation_step_outputs[0]
        img = torch.einsum('chw->hwc', out["img"]).detach().cpu().numpy() * 255
        pre_img = torch.einsum('chw->hwc', out["pre_img"]).detach().cpu().numpy() * 255
        mask = np.repeat(np.expand_dims(np.argmax(torch.einsum('chw->hwc', out["mask"]).detach().cpu().numpy(), 2),axis=-1),3,axis=2)* 255
        pre_mask = np.repeat(np.expand_dims(np.argmax(torch.einsum('chw->hwc', out["pre_mask"]).detach().cpu().numpy(), 2),axis=-1),3,axis=2)* 255

        output_lab = np.hstack([img, mask])
        output_pre = np.hstack([pre_img, pre_mask])
        output = np.vstack([output_lab, output_pre])


        if self.logger:
            self.logger.log_image(
                key='Generated result',
                images=[output],
                caption=[f'{self.current_epoch}_epoch'])

        self.validation_step_outputs.clear()





