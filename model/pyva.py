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
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        x = self.outLayer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x batch , 3 , h, w
        x = self.get_eff_features(x)
        return x



class CVT(nn.Module):
    def __init__(self,feat_dim,out_dim,feat_c):
        super(CVT,self).__init__()
        self.out_channels = int(out_dim[0]*out_dim[1])
        self.shape_transform = nn.Sequential(
            nn.Linear(int(feat_dim[0] * feat_dim[1]), self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU()
        )
        self.out_dim = out_dim
        self.feat_c = feat_c

        self.ft = nn.Sequential(
            nn.Conv2d(feat_c,feat_c,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(feat_c, feat_c, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )
        self.bt = nn.Sequential(
            nn.Conv2d(feat_c,feat_c,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(feat_c, feat_c, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )

        self.query_conv = nn.Conv2d(in_channels=feat_c, out_channels=feat_c // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=feat_c, out_channels=feat_c // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=feat_c, out_channels=feat_c, kernel_size=1)
        self.f_conv = nn.Conv2d(in_channels=feat_c * 2, out_channels=feat_c, kernel_size=3, stride=1, padding=1,
                                bias=True)

    def forward(self,img_feat):
        img_feat = img_feat.view(img_feat.size(0),img_feat.size(1),-1)
        img_feat = self.shape_transform(img_feat)

        img_feat = img_feat.view(img_feat.size(0),img_feat.size(1),self.out_dim[0],self.out_dim[1])
        f_feat = self.ft(img_feat)
        b_feat = self.bt(f_feat)

        B, C, H, W = img_feat.shape[0], img_feat.shape[1], img_feat.shape[2], img_feat.shape[3]
        query = self.query_conv(f_feat)
        key = self.key_conv(img_feat)
        value = self.value_conv(b_feat)  # B C H W
        value = value.view(B, C, -1)

        att_score = torch.bmm(key.view(B, self.feat_c // 8, -1).permute(0, 2, 1),
                              query.view(B, self.feat_c // 8, -1))  # B N N
        max_value, idx = att_score.max(dim=-1)
        idx = idx.unsqueeze(1).repeat(1, C, 1)
        max_value = max_value.view(B, 1, H, W)

        selected_value = torch.gather(value, 2, idx).view(B, C, H, W)
        front_res = torch.cat((f_feat, selected_value), dim=1)
        front_res = self.f_conv(front_res)
        front_res = front_res * max_value
        out = f_feat + front_res

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super(DecoderBlock, self).__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2):
        super(Decoder, self).__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y, x)

        return y


class BuildModel(nn.Module):
    def __init__(self,feat_c,feat_dim,out_dim,blocks,use_radar,radar_ch,Y):
        super(BuildModel,self).__init__()
        self.backbone = ImageBackbone(C=feat_c)
        self.cvt = CVT(feat_dim=feat_dim,out_dim=out_dim,feat_c=feat_c)
        self.decoder = Decoder(dim=feat_c,blocks=blocks)
        if use_radar:
            bev_feat_channel = blocks[-1] + radar_ch*Y
        else:
            bev_feat_channel = blocks[-1]

        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(bev_feat_channel, bev_feat_channel, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(bev_feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_feat_channel, 1, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(bev_feat_channel, bev_feat_channel, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(bev_feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_feat_channel, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(bev_feat_channel,bev_feat_channel, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(bev_feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_feat_channel, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.use_radar = use_radar

    def forward(self,img,radar,intrinsic):
        img_feat = self.backbone(img)
        bev_feat = self.cvt(img_feat)
        bev_feat = self.decoder(bev_feat)

        if self.use_radar:
            radar = radar.permute(0, 3, 1, 2)
            bev_feat = torch.cat((bev_feat,radar),dim=1)

        seg = self.segmentation_head(bev_feat)
        center = self.instance_center_head(bev_feat)
        offset = self.instance_offset_head(bev_feat)

        return seg,center,offset
