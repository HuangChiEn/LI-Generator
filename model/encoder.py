import torch
import clip
import torch.nn as nn
from torchvision.models import resnet18

def get_norm_by_name(norm_name, out_channel):
    if norm_name == "batch":
        return nn.BatchNorm2d(out_channel)
    if norm_name == "instance":
        return nn.InstanceNorm2d(out_channel)
    if norm_name == "none":
        return nn.Sequential()
    raise NotImplementedError("The norm name is not recognized %s" % (norm_name))


class Encoder(nn.Module):
    def __init__(self, clip_version, num_blocks, mask_in_dim, mask_out_dim, mask_norm_name):
        super(Encoder, self).__init__()
        self.clip_model, self._im_trfs = clip.load(clip_version)


        self.num_blocks = num_blocks
        input_dim = mask_in_dim

        self.mask_encoder = []

        self.mask_encoder.append(nn.Conv2d(input_dim, input_dim * 2, kernel_size=3, stride=2, padding=1))
        input_dim *= 2
        self.mask_encoder.append(get_norm_by_name(mask_norm_name, input_dim))
        for i in range(self.num_blocks - 2):
            self.mask_encoder.append(nn.Conv2d(input_dim, input_dim*2, kernel_size=3, stride=2, padding=1))
            input_dim *= 2
        self.mask_encoder.append(nn.Conv2d(input_dim, mask_out_dim, kernel_size=3, stride=2, padding=1))
        self.mask_encoder.append(get_norm_by_name(mask_norm_name, mask_out_dim))

        self.mask_encoder = nn.sequential(*self.mask_encoder)

    @property
    def im_trfs(self):
        return self._im_trfs

    @property
    def learnable_params(self):
        [{"name": "mask_encoder", "params": self.mask_encoder.parameters()}]

    def get_image_embeding(self, img):
        return self.clip_model.encode_image(img)
    def get_text_embeding(self, text):
        return self.clip_model.encode_text(text)
    def get_mask_embeding(self, mask):
        # for i in range(len(self.mask_encoder)):
        #     mask = self.mask_encoder[i](mask)
        #     print(mask.shape)
        msak = self.mask_encoder(mask)
        mask = torch.squeeze(mask)
        return mask
    def forward(self, image, mask):
        y = torch.cat((self.get_image_embeding(image), self.get_mask_embeding(mask)) , dim = 1)
        return y

if __name__ == '__main__':
    generator = Encoder("ViT-B/32",8,2,512,"batch")
    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2, 2, 224, 224)
    x = generator(x, y)
    print(x.shape)
    #(2,1024)
    #print(mask.shape)

