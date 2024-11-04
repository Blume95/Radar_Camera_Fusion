from torch import nn
import torch
from efficientnet_pytorch import EfficientNet
from einops import rearrange, repeat
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from tqdm import tqdm
import torchvision


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip

class ImageBackbone(nn.Module):
    def __init__(self, C: int, weight="efficientnet-b0",mid_channel=512):
        super(ImageBackbone, self).__init__()
        self.C = C
        self.trunk = EfficientNet.from_pretrained(weight)
        if weight == "efficientnet-b0":
            self.up1 = Up(320 + 112, mid_channel)
        elif weight == "efficientnet-b1":
            self.up1 = Up(320 + 112, mid_channel)
        elif weight == "efficientnet-b2":
            self.up1 = Up(352 + 120, mid_channel)
        elif weight == "efficientnet-b3":
            self.up1 = Up(384 + 136, mid_channel)
        elif weight == "efficientnet-b4":
            self.up1 = Up(448 + 160, mid_channel)
        else:
            raise NotImplemented

        self.outLayer = nn.Conv2d(mid_channel, self.C, kernel_size=1, padding=0)

    def get_eff_features(self, x: torch.Tensor) -> torch.Tensor:
        endpoints = dict()
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f'reduction_{len(endpoints) + 1}'] = prev_x
            prev_x = x

        endpoints[f'reduction_{len(endpoints) + 1}'] = x
        print(endpoints['reduction_5'].shape)
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        x = self.outLayer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x batch , 3 , h, w
        x = self.get_eff_features(x)
        return x


if __name__ == "__main__":
    # # test backbone
    backbone =  ImageBackbone(C=128,mid_channel=256)
    x = torch.rand((3,3,256,512))
    print(backbone(x).shape)
    # create_grid3d(2,32,8,16)