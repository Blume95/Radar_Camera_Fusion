import yaml
import torch
import torch.nn.functional as F


def save_cfg(dict_data,name):
    yaml.Dumper.ignore_aliases = lambda *args: True
    with open(f"../cfg/{name}.yml","w") as f:
        yaml.dump(dict_data,f,default_flow_style=False,sort_keys=False)
    f.close()


def read_cfg(name):
    with open(name) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    f.close()

    return data


def build_dataset(cfg):
    if cfg.dataset == "tu_delft":
        from dataset.vod import dataloaders
        train_loader,val_loader = dataloaders(path=cfg.dataset_path,grid=cfg.grid,
                    nworkers=cfg.workers,batch_size=cfg.batch_size,
                    data_aug_conf=cfg.data_aug_conf,useRadar=cfg.use_Radar,
                    useCamera=cfg.use_Cam,data_aug=cfg.data_aug)
        return train_loader, val_loader
    elif cfg.dataset == "expleo":
        train_loader, val_loader =None,None
        pass


def build_model(cfg):
    if cfg.model == "Bili":
        from model.bili import BuildModel
        model_ = BuildModel(Z=cfg.voxel_z,
                         Y=cfg.voxel_y,
                         X=cfg.voxel_x,x_meter=cfg.x_meter,y_meter=cfg.y_meter,z_meter=cfg.z_meter,
                           img_h=cfg.final_hw[0], img_w=cfg.final_hw[1],
                           feat_c=cfg.img_feat_c,bev_ch=cfg.out_ch,radar_ch=cfg.radar_channel,use_radar=cfg.use_Radar)
        return model_

    if cfg.model == "PYVA":
        from model.pyva import BuildModel
        model_ = BuildModel(feat_c=cfg.img_feat_c,feat_dim=(int(cfg.final_hw[0]//16),int(cfg.final_hw[1]//16)),
                           out_dim=(int(cfg.voxel_z//16),int(cfg.voxel_x//16)),blocks=cfg.blocks,
                           use_radar=cfg.use_Radar,radar_ch=cfg.radar_channel,Y=cfg.voxel_y)
        return model_
    if cfg.model == "CVT":
        from model.cvt import BuildModel
        model_ = BuildModel(feat_c=cfg.img_feat_c,depth=cfg.z_meter,img_h=cfg.final_hw[0], img_w=cfg.final_hw[1],
                            Z=cfg.voxel_z,Y=cfg.voxel_y,X=cfg.voxel_x,x_meter=cfg.x_meter,y_meter=cfg.y_meter,z_meter=cfg.z_meter,bev_dim=cfg.out_ch)
        return model_

    else:
        raise NotImplementedError




def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    EPS = 1e-6
    for (a,b) in zip(x.size(), mask.size()):
        # if not b==1:
        assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer/denom
    return mean

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = reduce_masked_mean(loss, valid)
        return loss


def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = reduce_masked_mean(mse_loss, pos_mask*valid)
    neg_loss = reduce_masked_mean(mse_loss, neg_mask*valid)
    loss = (pos_loss + neg_loss)*0.5
    return loss






if __name__ =="__main__":
    # from argparse import Namespace
    # import torchvision
    # cfgs = read_cfg(name="tu_delft_bili_29_10")
    # cfgs = Namespace(**cfgs)
    # train,val = build_dataset(cfgs)
    # train_iter_loader = iter(train)
    # val_iter_loader = iter(val)
    # for val_d in val_iter_loader:
    #     img = val_d['image']
    #     intrinsic = val_d['intrinsic']
    #     seg_bev = val_d['seg_bev']
    #     offset = val_d['offset']
    #     center = val_d['center']
    #     torchvision.utils.save_image(img[0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/val_d.png")
    #
    #     torchvision.utils.save_image(seg_bev[0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/val_seg.png")
    #     torchvision.utils.save_image(offset[0, 0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/val_offset_0.png")
    #     torchvision.utils.save_image(offset[0, 1], "/home/jing/Downloads/Radar_Camera_Fusion/vis/val_offset_1.png")
    #     torchvision.utils.save_image(center[0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/val_center.png")
    #     break
    # for train_d in train_iter_loader:
    #     img = train_d['image']
    #     intrinsic = train_d['intrinsic']
    #     seg_bev = train_d['seg_bev']
    #     offset = train_d['offset']
    #     center = train_d['center']
    #     torchvision.utils.save_image(img[0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/train_d.png")
    #
    #     torchvision.utils.save_image(seg_bev[0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/train_seg.png")
    #     torchvision.utils.save_image(offset[0, 0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/train_offset_0.png")
    #     torchvision.utils.save_image(offset[0, 1], "/home/jing/Downloads/Radar_Camera_Fusion/vis/train_offset_1.png")
    #     torchvision.utils.save_image(center[0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/train_center.png")
    #     break

    # different training cfg
    # Bili tu delf
    org_hw = [1216, 1936]

    xbound = [-8.0, 8.0, 0.1]
    ybound = [-4.0, 4.0, 1.0]
    zbound = [0, 32, 0.1]
    dbound = [3.0, 43.0, 1.0]

    voxel_x = int((xbound[1] - xbound[0])/xbound[2])
    voxel_y = int((ybound[1] - ybound[0])/ybound[2])
    voxel_z = int((zbound[1] - zbound[0])/zbound[2])

    x_meter = xbound[1] - xbound[0]
    y_meter = ybound[1] - ybound[0]
    z_meter = zbound[1] - zbound[0]

    grid = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    bev_size = [int((grid['zbound'][1] - grid['zbound'][0]) / grid['zbound'][2]),
                int((grid['xbound'][1] - grid['xbound'][0]) / grid['xbound'][2]),
                int((grid['ybound'][1] - grid['ybound'][0]) / grid['ybound'][2])]
    rand_crop_and_resize = True
    res_scale = 0.5
    blocks = [128, 128, 64]

    XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = (grid['xbound'][0],grid['xbound'][1]
                                          ,grid['ybound'][0],grid['ybound'][1],
                                          grid['zbound'][0],grid['zbound'][1])

    bounds = [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]

    final_dim = [int(512 * res_scale), int(1024 * res_scale)]
    print('resolution:', final_dim)

    if rand_crop_and_resize:
        resize_lim = [0.8, 1.2]
        crop_offset = int(final_dim[0] * (1 - resize_lim[0]))
    else:
        resize_lim = [1.0, 1.0]
        crop_offset = 0
    data_aug_conf = {
        'crop_offset': crop_offset,
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'H': org_hw[0], 'W': org_hw[1],
    }

    out_channel = 1
    use_Radar = False
    use_Cam = True
    radar_channel = 3
    cam_channel = 64

    training_parameters = {
        "dataset":"tu_delft",
        "dataset_path":"/home/jing/Downloads/view_of_delft_PUBLIC/",
        "model":"Bili",
        "batch_size": 8,
        "workers": 4,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "nepochs": 200,
        "val_step": 200,
        "img_feat_c": 512,
        "out_ch":256,
        "grad_acc":4,
        "max_iters":86000,
        "grid": grid,
        "final_hw": final_dim,
        "org_hw": org_hw,
        "cam_channel": cam_channel,
        "radar_channel": radar_channel,
        "use_Radar": use_Radar,
        "use_Cam": use_Cam,
        "data_aug_conf": data_aug_conf,
        "out_channel": out_channel,
        "voxel_x":voxel_x,
        "voxel_y": voxel_y,
        "voxel_z": voxel_z,
        "x_meter": x_meter,
        "y_meter": y_meter,
        "z_meter": z_meter,
        "bounds":bounds,
    }
    save_cfg(training_parameters,"tu_delft_bili_29_10")
