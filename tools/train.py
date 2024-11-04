from tools.tool import (read_cfg, build_model, build_dataset,requires_grad,
                        fetch_optimizer,balanced_mse_loss,SimpleLoss,
                        reduce_masked_mean)
from argparse import Namespace
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import socket
import torchvision
from tqdm import tqdm
# from torchviz import make_dot, make_dot_from_trace

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
datetime_now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')



def train(cfg,experiment_name):
    device_ids = [0, 1]
    assert(cfg.batch_size % len(device_ids) == 0) # batch size must be divisible by number of gpus
    if cfg.grad_acc > 1:
        print('effective batch size:', cfg.batch_size*cfg.grad_acc)
    device = 'cuda:%d' % device_ids[0]

    train_loader, val_loader = build_dataset(cfg)
    train_iter_loader = iter(train_loader)

    global_step = 0
    total_intersect = 0
    total_union = 0

    model = build_model(cfg)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model_parameters = list(model.parameters())
    optimizer, scheduler = fetch_optimizer(cfg.lr, cfg.weight_decay, 1e-8, cfg.max_iters, model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params', total_params)
    requires_grad(model_parameters, True)
    model.train()

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        "runs", current_time + "_" + experiment_name
    )
    writer = SummaryWriter(log_dir=log_dir)
    weights_dir = f"runs/{experiment_name}"
    os.makedirs(weights_dir, exist_ok=True)

    seg_loss_fn = SimpleLoss(2).to(device)


    # training loop
    while global_step < cfg.max_iters:
        print(f"Global Step {global_step}")
        global_step += 1

        for internal_step in range(cfg.grad_acc):
            try:
                sample = next(train_iter_loader)
            except StopIteration:
                train_iter_loader = iter(train_loader)
                sample = next(train_iter_loader)

            total_loss = torch.tensor(0.0, requires_grad=True).to(device)

            img = sample['image'].to(device)
            intrinsic = sample['intrinsic'].to(device)
            seg_bev = sample['seg_bev'].to(device)
            offset = sample['offset'].to(device)
            center = sample['center'].to(device)
            radar = sample['radar_features'].to(device)

            # raw_e, feat_e, seg_e, center_e, offset_e
            seg_bev_pred,center_bev_pred,offset_bev_pred = model(img,radar,intrinsic)

            valid_bev_g = torch.ones_like(seg_bev,dtype=torch.float32)
            ce_loss = seg_loss_fn(seg_bev_pred, seg_bev, valid_bev_g)
            center_loss = balanced_mse_loss(center_bev_pred, center)
            offset_loss = torch.abs(offset_bev_pred - offset).sum(dim=1, keepdim=True)
            offset_loss = reduce_masked_mean(offset_loss, seg_bev * valid_bev_g)

            ce_factor = 1 / torch.exp(model.module.ce_weight)
            ce_loss = 10.0 * ce_loss * ce_factor
            ce_uncertainty_loss = 0.5 * model.module.ce_weight

            center_factor = 1 / (2 * torch.exp(model.module.center_weight))
            center_loss = center_factor * center_loss
            center_uncertainty_loss = 0.5 * model.module.center_weight

            offset_factor = 1 / (2 * torch.exp(model.module.offset_weight))
            offset_loss = offset_factor * offset_loss
            offset_uncertainty_loss = 0.5 * model.module.offset_weight

            total_loss += ce_loss
            total_loss += center_loss
            total_loss += offset_loss
            total_loss += ce_uncertainty_loss
            total_loss += center_uncertainty_loss
            total_loss += offset_uncertainty_loss

            total_loss.backward()

            pred_mask = (seg_bev_pred>0)

            tgt = seg_bev.bool()
            total_intersect += (pred_mask & tgt).sum().float().item()
            total_union += (pred_mask | tgt).sum().float().item()


        if global_step % 10 == 0 and device=="cuda:0":
            writer.add_scalar('train/iou', total_intersect / (total_union+1e-7), global_step)
            writer.add_scalar('train/center_loss',center_loss,global_step)
            writer.add_scalar('train/ce_loss',ce_loss,global_step)
            writer.add_scalar('train/offset_loss',offset_loss,global_step)
            writer.add_scalar('train/total_loss',total_loss,global_step)
            total_intersect = 0
            total_union = 0

        # if global_step % grad_acc == 0:
        torch.nn.utils.clip_grad_norm_(model_parameters, 5.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()



        if global_step % 100 == 0 and device=="cuda:0":
            torch.cuda.empty_cache()
            model.eval()
            mname = f"{weights_dir}/weights.pt"
            val_iter_loader = iter(val_loader)
            total_intersect_val = 0
            total_union_val = 0

            for d in tqdm(val_iter_loader):
                with torch.no_grad():
                    img = d['image'].to(device)
                    intrinsic = d['intrinsic'].to(device)
                    seg_bev = d['seg_bev'].to(device)
                    radar = d['radar_features'].to(device)

                    seg_bev_pred, center_bev_pred, offset_bev_pred = model(img,radar,intrinsic)
                    pred_mask = (seg_bev_pred > 0)
                    tgt = seg_bev.bool()
                    total_intersect_val += (pred_mask & tgt).sum().float().item()
                    total_union_val += (pred_mask | tgt).sum().float().item()
            writer.add_scalar("val/iou",total_intersect_val/(total_union_val+1e-7), global_step)
            torch.save(model.module.state_dict(), mname)

            model.train()




if __name__ == "__main__":
    from glob import glob
    yaml_files = glob(f"/home/jing/Downloads/Radar_Camera_Fusion/cfg/*.yml")
    for files in yaml_files:
        exp_name = files.split("/")[-1].replace(".yml",'')
        cfgs = read_cfg(name=files)
        cfgs = Namespace(**cfgs)
        train(cfgs,exp_name)
