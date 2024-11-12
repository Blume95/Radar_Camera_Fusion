from typing import Tuple, Any

from shapely.examples.dissolve import points
from torch import nn
from efficientnet_pytorch import EfficientNet
import torch
from torchvision.models.resnet import resnet18



def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

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

class CamEncode(nn.Module):
    def __init__(self, D: int, C: int,weight="efficientnet-b0", mid_channels=512):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.trunk = EfficientNet.from_pretrained(weight)

        if weight == "efficientnet-b0":
            self.up1 = Up(320 + 112, mid_channels)
        elif weight == "efficientnet-b1":
            self.up1 = Up(320 + 112, mid_channels)
        elif weight == "efficientnet-b2":
            self.up1 = Up(352 + 120, mid_channels)
        elif weight == "efficientnet-b3":
            self.up1 = Up(384 + 136, mid_channels)
        elif weight == "efficientnet-b4":
            self.up1 = Up(448 + 160, mid_channels)
        else:
            raise NotImplemented

        self.depthnet = nn.Conv2d(mid_channels, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        return x.softmax(dim=1)

    def get_depth_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.get_eff_depth(x)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        return new_x

    def get_eff_depth(self, x: torch.Tensor) -> torch.Tensor:
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
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x batch , 3 , h, w
        x = self.get_depth_feat(x)
        #x: batch, C, D H W
        return x

class BuildModel(nn.Module):
    def __init__(self,input_img_hw,feature_downsize,grid_conf,feat_c,use_radar,radar_ch,voxel_y,outC=64):
        super(BuildModel, self).__init__()
        self.input_img_hw = input_img_hw
        self.feature_downsize = feature_downsize
        self.grid_conf = grid_conf
        self.use_radar = use_radar
        self.radar_ch = radar_ch
        self.outC = outC

        self.frustum =  self.creat_frustum_one_cam()

        self.encoder = CamEncode(D=self.frustum.shape[0],C=feat_c,weight="efficientnet-b0")
        self.decoder = BevEncode(inC=feat_c, outC=outC)
        if use_radar:
            self.decoder = BevEncode(inC=feat_c+radar_ch*voxel_y,outC=outC)

        self.nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [self.grid_conf['xbound'],
                                                                           self.grid_conf['ybound'],
                                                                           self.grid_conf['zbound']]])
        self.dx = torch.Tensor([row[2] for row in [self.grid_conf['xbound'],
                                                   self.grid_conf['ybound'],
                                                   self.grid_conf['zbound']]])
        self.bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [self.grid_conf['xbound'],
                                                                  self.grid_conf['ybound'],
                                                                  self.grid_conf['zbound']]])

        # self.dx = nn.Parameter(self.dx, requires_grad=False)
        # self.bx = nn.Parameter(self.bx, requires_grad=False)
        # self.nx = nn.Parameter(self.nx, requires_grad=False)

        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(outC, outC, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC, 1, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(outC, outC, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(outC,outC, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        self.use_quickcumsum = True

    def creat_frustum_one_cam(self):
        input_img_h, input_img_w = self.input_img_hw[0], self.input_img_hw[1]
        feat_h, feat_w =  input_img_h // self.feature_downsize, input_img_w // self.feature_downsize
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, feat_h, feat_w)
        D, _, _ = ds.shape
        xs = torch.linspace(0, input_img_w - 1, feat_w, dtype=torch.float).view(1, 1, feat_w).expand(D, feat_h, feat_w)
        ys = torch.linspace(0, input_img_h - 1, feat_h, dtype=torch.float).view(1, feat_h, 1).expand(D, feat_h, feat_w)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def apply_transformation(self,intrins: torch.Tensor)-> torch.Tensor:
        B = intrins.shape[0]
        points = self.frustum.unsqueeze(0).repeat(B,1,1,1,1)
        # points = self.frustum - post_trans.view(B, 1, 1, 1, 3)
        # points = torch.inverse(post_rots).view(B, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat((points[:, :, :, :, :2] * points[:, :, :, :, 2:3],
                            points[:, :, :, :, 2:3]), 4).unsqueeze(-1)
        points = torch.inverse(intrins).view(B, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        return points

    def get_cam_feats(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.permute(0, 2, 3, 4, 1)

        return x

    def voxel_splat(self, look_up_table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        Nprime = B * D * H * W
        x = x.reshape(Nprime, C)
        self.bx =self.bx.to(x.device)
        self.dx =self.dx.to(x.device)
        self.nx =self.nx.to(x.device)

        look_up_table = ((look_up_table - (self.bx - self.dx / 2.)) / self.dx).long()
        look_up_table = look_up_table.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        look_up_table = torch.cat((look_up_table, batch_ix), 1)

        kept = (look_up_table[:, 0] >= 0) & (look_up_table[:, 0] < self.nx[0]) & (look_up_table[:, 1] >= 0) & (
                look_up_table[:, 1] < self.nx[1]) & (look_up_table[:, 2] >= 0) & (look_up_table[:, 2] < self.nx[2])
        x = x[kept]
        look_up_table = look_up_table[kept]

        ranks = look_up_table[:, 0] * (self.nx[1] * self.nx[2] * B) + look_up_table[:, 1] * (
                self.nx[2] * B) + look_up_table[:, 2] * B + look_up_table[:, 3]
        sorts = ranks.argsort()
        x, look_up_table, ranks = x[sorts], look_up_table[sorts], ranks[sorts]

        if not self.use_quickcumsum:
            x, look_up_table = cumsum_trick(x, look_up_table, ranks)
        else:
            x, look_up_table = QuickCumsum.apply(x, look_up_table, ranks)

        final = torch.zeros((B, C, self.nx[1], self.nx[2], self.nx[0]), device=x.device)
        final[look_up_table[:, 3], :, look_up_table[:, 1], look_up_table[:, 2], look_up_table[:, 0]] = x

        final = torch.cat(final.unbind(dim=2), 1)
        return final

    def forward(self, x: torch.Tensor, radar_bev: torch.Tensor,intrins: torch.Tensor) -> tuple[Any, Any, Any]:
        img_feat = self.get_cam_feats(x)
        geom = self.apply_transformation(intrins)
        bev_feat =  self.voxel_splat(geom, img_feat)

        if self.use_radar:
            radar_bev = radar_bev.permute(0, 3, 1, 2)
            bev_feat = torch.cat((bev_feat, radar_bev), dim=1)

        bev_feat = self.decoder(bev_feat)

        seg = self.segmentation_head(bev_feat)
        center = self.instance_center_head(bev_feat)
        offset = self.instance_offset_head(bev_feat)

        return seg,center,offset




class BevEncode(nn.Module):
    def __init__(self, inC: int, outC: int):
        super(BevEncode, self).__init__()
        trunk = resnet18(weights=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.up1(x, x1)
        x = self.up2(x)
        return x

if __name__ == '__main__':
    encoder = CamEncode(D=32,C=64,weight="efficientnet-b0")
    img = torch.rand(3, 3, 256, 256)
    feat = encoder(img)
    print(feat.shape)