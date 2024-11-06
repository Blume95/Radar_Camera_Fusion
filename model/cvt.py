from torch import nn
import torch
from efficientnet_pytorch import EfficientNet
from einops import rearrange, repeat
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from tqdm import tqdm
import torchvision
from torchvision.models.resnet import Bottleneck



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

    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz

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
    def __init__(self, C: int,weight="efficientnet-b0",mid_channel=512):
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

    def get_eff_features(self, x: torch.Tensor):
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
        return x, endpoints['reduction_3'] # x: b 512 h/16 w/16, 3_re: b 40 h/8 w/8

    def forward(self, x: torch.Tensor):
        # x batch , 3 , h, w
        x,pre_x = self.get_eff_features(x)

        return x,pre_x


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super(CrossAttention, self).__init__()
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.get_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.get_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.get_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.pre_norm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.post_norm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """

        :param q: (b,d,H,W)
        :param k: (b,d,h,w)
        :param v: (b,d,h,w)
        :param skip: (b,d,H,W)
        :return:
        """

        _, _, H, W = q.shape

        q = rearrange(q, 'b d H W -> b (H W) d')
        k = rearrange(k, 'b d h w -> b (h w) d')
        v = rearrange(v, 'b d h w -> b (h w) d')

        q = self.get_q(q)
        k = self.get_k(k)
        v = self.get_v(v)

        # multi-head
        q = rearrange(q, 'b ... (m d) -> (m b) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (m b) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (m b) ... d', m=self.heads, d=self.dim_head)

        dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)
        att = dot.softmax(dim=-1)

        v_selected = torch.einsum('b Q K, b K d -> b Q d', att, v)
        v_selected = rearrange(v_selected, '(m b) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        z = self.proj(v_selected)
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.pre_norm(z)
        z = z + self.mlp(z)
        z = self.post_norm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class CrossViewAttention(nn.Module):
    def __init__(self,depth,img_h,img_w,feat_h,feat_w,Z,Y,X,bev_dim,feat_dim,heads,dim_head,qkv_bias):
        super(CrossViewAttention,self).__init__()
        self.image_plane = self.creat_img_xyz(depth,feat_h,feat_w)
        self.bev_plane = self.creat_bev_xyz(Z,Y,X)
        self.sx = feat_w / img_w
        self.sy = feat_h / img_h
        self.X = X
        self.Y = Y
        self.Z = Z
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.depth = depth
        self.bev_embed = nn.Linear(3*Y,bev_dim)
        self.img_embed = nn.Linear(3*depth,bev_dim)
        self.value_proj = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, bev_dim, 1, bias=False))
        self.key_proj = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, bev_dim, 1, bias=False))
        self.cross_attention = CrossAttention(bev_dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias)

    def creat_img_xyz(self,depth,feat_h,feat_w):
        img_plane = create_grid3d(B=1,Z=depth,Y=feat_h,X=feat_w) # x y z -> u v d
        img_plane[...,0] *= (feat_w*depth)
        img_plane[...,1] *= (feat_h*depth)
        img_plane[...,2] *= depth # b n 3 n=

        return img_plane

    def creat_bev_xyz(self,Z,Y,X):
        bev_plane = create_grid3d(B=1,Z=Z,Y=Y,X=X)
        bev_plane[...,0] -= 0.5
        bev_plane[...,1] -= 0.5

        bev_plane[...,0] *= X
        bev_plane[...,1] *= Y
        bev_plane[...,2] *= Z # b n 3

        return bev_plane

    def forward(self,img_feat,intrinsic,bev_feat):
        intrinsic[:,0,:] *= self.sx
        intrinsic[:,1,:] *= self.sy
        inv_intrinsic = intrinsic.inverse()


        image_plane = repeat(self.image_plane,'b n c -> (repeat b) n c',repeat=img_feat.shape[0])
        # print(f"image_plane: {image_plane.shape}")
        image_plane = torch.transpose(image_plane,1,2)
        image_plane = image_plane.to(img_feat.device)
        xyz = torch.transpose(torch.matmul(inv_intrinsic,image_plane),1,2) # b n 3
        # depth_feat = rearrange(depth_feat,'b c h w -> b (c h w)')
        # depth_feat = depth_feat[:,:,None]
        # print(f"xyz;{xyz.shape}")
        # print(f"depth_feat;{depth_feat.shape}")
        # xyz_depth = torch.cat((xyz,depth_feat),dim=2)
        xyz = rearrange(xyz,'b (d h w) c -> b d h w c',b=img_feat.shape[0],d=self.depth,
                              h=self.feat_h,w=self.feat_w,c=3)
        xyz_depth = rearrange(xyz, 'b d h w c -> b h w (c d)')
        img_pe = self.img_embed(xyz_depth).permute(0,3,1,2)
        # print(f"img_feat.shape: {img_feat.shape}")
        key_feat = self.key_proj(img_feat)

        key = img_pe + key_feat

        bev_plane = repeat(self.bev_plane,'b n c -> (repeat b) n c',repeat=img_feat.shape[0])
        bev_plane = rearrange(bev_plane,'b (z y x) c -> b z y x c',b=img_feat.shape[0],z=self.Z,y=self.Y,x=self.X,c=3)
        bev_plane = rearrange(bev_plane,'b d h w c -> b d w (c h)')
        bev_plane = bev_plane.to(img_feat.device)
        bev_pe = self.bev_embed(bev_plane).permute(0,3,1,2)


        query = bev_pe + bev_feat

        value =  self.value_proj(img_feat)
        return self.cross_attention(query,value,key)


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



class BuildModel(nn.Module):
    def __init__(self,feat_c,depth,img_h,img_w,Z,Y,X,bev_dim):
        super(BuildModel, self).__init__()
        self.backbone = ImageBackbone(C=feat_c,weight="efficientnet-b0")
        self.bev_feat = nn.Parameter(torch.rand(1,bev_dim,Z,X),requires_grad=True)
        self.cva0 = CrossViewAttention(depth,img_h,img_w,img_h //16,img_w//16,Z,Y,X,bev_dim,feat_c,heads=4,dim_head=32,qkv_bias=True)
        self.cva1 = CrossViewAttention(depth,img_h,img_w,img_h //8,img_w//8,Z,Y,X,bev_dim,feat_dim=40,heads=4,dim_head=32,qkv_bias=True)
        self.layer0 = Bottleneck(bev_dim, bev_dim // 4)
        self.layer1 = Bottleneck(bev_dim, bev_dim // 4)
        self.decoder = Decoder(bev_dim,1)

        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)


    def forward(self,img,radar,intrinsic):
        features, pre_reduced_features = self.backbone(img)
        bev_feat = self.bev_feat.repeat((img.shape[0],1,1,1))

        x = self.cva0(features,intrinsic,bev_feat)
        x = self.layer0(x)
        x = self.cva1(pre_reduced_features,intrinsic,x)
        x = self.layer1(x)
        return self.decoder(x)



if __name__ == "__main__":
    # # test backbone
    backbone =  ImageBackbone(C=128,mid_channel=256)
    x = torch.rand((3,3,256,512))
    print(backbone(x).shape)
    # create_grid3d(2,32,8,16)