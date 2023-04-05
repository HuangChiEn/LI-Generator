import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sp_norm


def get_norm_by_name(norm_name, out_channel):
    if norm_name == "batch":
        return nn.BatchNorm2d(out_channel)
    if norm_name == "instance":
        return nn.InstanceNorm2d(out_channel)
    if norm_name == "none":
        return nn.Sequential()
    raise NotImplementedError("The norm name is not recognized %s" % (norm_name))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, num_blocks, norm_name, input_shape, input_dim, no_masks, num_mask_channels):
        super(Generator, self).__init__()
        self.num_blocks = num_blocks
        # self.noise_shape = config_G["noise_shape"]
        self.input_shape = input_shape
        self.input_dim = input_dim
        # self.noise_init_dim = config_G["noise_dim"]
        self.norm_name = norm_name
        self.no_masks = no_masks
        self.num_mask_channels = num_mask_channels
        num_of_channels = ([8, 8, 8, 8, 8, 8, 8, 4, 2, 1] * 32)[-self.num_blocks-1:]

        self.body, self.rgb_converters = nn.ModuleList([]), nn.ModuleList([])
        self.first_linear = nn.ConvTranspose2d(self.input_dim, num_of_channels[0], self.input_shape)
        for i in range(self.num_blocks):
            cur_block = G_block(num_of_channels[i], num_of_channels[i+1], self.norm_name, i==0)
            cur_rgb  = sp_norm(nn.Conv2d(num_of_channels[i+1], 3, (3, 3), padding=(1, 1), bias=True))
            self.body.append(cur_block)
            self.rgb_converters.append(cur_rgb)
        if not self.no_masks:
            self.mask_converter = nn.Conv2d(num_of_channels[i+1], self.num_mask_channels, 3, padding=1, bias=True)
        print("Created Generator with %d parameters" % (sum(p.numel() for p in self.parameters())))

    def forward(self, z, get_feat=False):
        output = dict()
        ans_images = list()
        ans_feat = list()
        x = self.first_linear(z)
        for i in range(self.num_blocks):
            print(x.shape)
            x = self.body[i](x)
            im = torch.tanh(self.rgb_converters[i](x))
            ans_images.append(im)
            ans_feat.append(torch.tanh(x))
        output["images"] = ans_images

        if get_feat:
             output["features"] = ans_feat
        if not self.no_masks:
            mask = self.mask_converter(x)
            mask = F.softmax(mask, dim=1)
        return ans_images, mask

class G_block(nn.Module):
    def __init__(self, in_channel, out_channel, norm_name, is_first):
        super(G_block, self).__init__()
        middle_channel = min(in_channel, out_channel)
        self.ups = nn.Upsample(scale_factor=2) if not is_first else torch.nn.Identity()
        self.activ = nn.LeakyReLU(0.2)
        self.conv1 = sp_norm(nn.Conv2d(in_channel,  middle_channel, 3, padding=1))
        self.conv2 = sp_norm(nn.Conv2d(middle_channel, out_channel, 3, padding=1))
        self.norm1 = get_norm_by_name(norm_name, in_channel)
        self.norm2 = get_norm_by_name(norm_name, middle_channel)
        self.conv_sc = sp_norm(nn.Conv2d(in_channel, out_channel, (1, 1), bias=False))

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.activ(x)
        x = self.ups(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv2(x)
        h = self.ups(h)
        h = self.conv_sc(h)
        return h + x

if __name__ == '__main__':
    generator = Generator(10,"batch",1,512,False,2)
    x = torch.rand(2,512,1,1)
    x, mask = generator(x)
    print(x[-1].shape)
    print(mask.shape)