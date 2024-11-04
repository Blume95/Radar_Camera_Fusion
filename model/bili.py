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
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        x = self.outLayer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x batch , 3 , h, w
        x = self.get_eff_features(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )


    def forward(self, x):
        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])


        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        return segmentation_output,instance_center_output,instance_offset_output



class BiliSample(nn.Module):
    def __init__(self,Z,Y,X,x_meter,y_meter,z_meter,img_h, img_w,feat_c,out_ch):
        super(BiliSample,self).__init__()
        self.xyz_cam = create_grid3d(1,Z,Y,X)
        self.xyz_cam[...,0] *= x_meter
        self.xyz_cam[...,1] *= y_meter
        self.xyz_cam[...,2] *= z_meter
        self.z = Z
        self.y = Y
        self.x = X
        self.img_h = img_h
        self.img_w = img_w
        self.reduce_feature_channel = nn.Conv2d(feat_c*Y,out_ch, kernel_size=1, padding=0)

    def forward(self,img_feat,pix_T_cam):
        b,c,h,w = img_feat.shape
        xyz_cam = repeat(self.xyz_cam,'b n c -> (repeat b) n c',repeat=b,b=1,c=3)
        xyz_cam = torch.transpose(xyz_cam,1,2)
        xyz_cam = xyz_cam.to(img_feat.device)
        xyz_pixel = torch.transpose(torch.matmul(pix_T_cam,xyz_cam),1,2)
        xyz_pixel[...,:2] = xyz_pixel[...,:2] / (xyz_pixel[...,2:3] + 1e-8)
        # print(xyz_pixel)


        uvd = xyz_pixel
        uvd[...,0] = uvd[...,0] / self.img_w
        uvd[...,1] = uvd[...,1] / self.img_h
        uvd[...,2] = 0


        uvd = uvd.reshape(shape=(b,self.z,self.y,self.x,3))
        # print(uvd[...,:2])

        valid_mask = (uvd[...,0] >=0) & (uvd[...,1] >=0) & (uvd[...,0]<=1) & (uvd[...,1]<=1) # b z y x
        valid_mask = repeat(valid_mask,'b d h w -> b c d h w', c=c)
        
        uvd[...,:2] = uvd[...,:2]*2-1
        # print(uvd)
        img_feat = img_feat.unsqueeze(2)
        # print(uvd)
        bev_feat = F.grid_sample(img_feat,uvd,align_corners=False) # b c d h w
        bev_feat[~valid_mask] = 0

        bev_features = rearrange(bev_feat,'b c d h w -> b (c h) d w')
        # torchvision.utils.save_image(bev_features[0, 1], "/home/jing/Downloads/Radar_Camera_Fusion/vis/bev_features.png")

        bev_features = self.reduce_feature_channel(bev_features)
        # print("-" * 80)

        return bev_features


class BuildModel(nn.Module):
    def __init__(self,Z,Y,X,x_meter,y_meter,z_meter,img_h, img_w,feat_c,bev_ch,use_radar,radar_ch=None):
        super(BuildModel,self).__init__()
        self.img_encoder = ImageBackbone(C=feat_c, weight="efficientnet-b0")
        self.sampling = BiliSample(Z,Y,X,x_meter,y_meter,z_meter,img_h, img_w,feat_c,bev_ch)
        self.bev_decoder = Decoder(bev_ch,1)
        if use_radar:
            self.bev_decoder = Decoder(bev_ch+radar_ch*Y, 1)

        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.use_radar = use_radar

    def forward(self,imgs,radar,pix_T_cam):
        # print(imgs.shape)
        img_feat = self.img_encoder(imgs)
        # print(img_feat.shape)
        bev_feat = self.sampling(img_feat,pix_T_cam)
        # print(bev_feat.shape)
        if self.use_radar:
            radar = radar.permute(0, 3, 1, 2)
            bev_feat =  torch.cat((bev_feat,radar),dim=1)

        return self.bev_decoder(bev_feat)


def create_grid3d(B,Z,Y,X,device="cuda"):
    grid_z = torch.linspace(0.0, 1.0, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, 1.0, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, 1.0, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])

    x = x-0.5
    y = y-0.5


    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz


if __name__ == "__main__":
    # # test backbone
    backbone =  ImageBackbone(C=128,mid_channel=256)
    x = torch.rand((3,3,256,512))
    print(backbone(x).shape)
    # create_grid3d(2,32,8,16)
