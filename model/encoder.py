import torch
import clip
import torch.nn as nn
from torchvision.models import resnet18

class Encoder(nn.Module):
    def __init__(self, clip_version):
        super(Encoder, self).__init__()
        self.clip_model, _ = clip.load(clip_version)
        self.mask_encoder = resnet18()
    def get_image_embeding(self, img):
        return self.clip_model.encode_image(img)
    def get_text_embeding(self, test):
        return self.clip_model.encode_text(test)
    def get_mask_embeding(self, mask):
        return self.mask_encoder(mask)
    def forward(self, image, mask):
        return self.get_image_embeding(image), self.get_mask_embeding(mask)

if __name__ == '__main__':
    generator = Encoder("ViT-B/32")
    x = torch.rand(2, 3, 224, 224)
    y = torch.rand(2, 3, 224, 224)
    x, mask = generator(x, y)
    print(x.shape)
    print(mask.shape)

