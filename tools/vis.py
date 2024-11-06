from tools.tool import read_cfg,build_model, build_dataset
from argparse import Namespace
import torch
from tqdm import tqdm
import numpy as np
import cv2
import os

esp = 1e-8

def visualization(cfgs,exp_name):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfgs)
    model.load_state_dict(
        torch.load(f"/home/jing/Downloads/weigths/runs_backup/runs/{exp_name}/weights.pt", map_location=device))
    vis_folder = f"/home/jing/Downloads/weigths/runs_backup/runs/{exp_name}/visualization"
    os.makedirs(vis_folder, exist_ok=True)

    model.to(device)
    model.eval()

    cfgs.batch_size = 2

    _, val_loader = build_dataset(cfgs)
    val_iter_loader = iter(val_loader)

    total_intersect = 0
    total_union = 0
    total_intersect_5 = 0
    total_union_5 = 0
    total_intersect_10 = 0
    total_union_10 = 0

    fn = 0
    fp = 0
    tp = 0
    tn = 0

    for d in tqdm(val_iter_loader):
        with torch.no_grad():
            img = d['image'].to(device)
            intrinsic = d['intrinsic'].to(device)
            seg_bev = d['seg_bev'].to(device)
            radar = d['radar_features'].to(device)
            img_name = d['img_name']

            seg_bev_pred, center_bev_pred, offset_bev_pred = model(img, radar, intrinsic)
            # print(seg_bev_pred.shape)
            # print(seg_bev.shape)
            # break
            pred_mask = (seg_bev_pred > 0)

            tgt = seg_bev.bool()
            total_intersect += (pred_mask & tgt).sum().float().item()
            total_union += (pred_mask | tgt).sum().float().item()
            total_intersect_5 += (pred_mask[:,:,:50,:] & tgt[:,:,:50,:]).sum().float().item()
            total_union_5 += (pred_mask[:,:,:50,:] | tgt[:,:,:50,:]).sum().float().item()
            total_intersect_10 += (pred_mask[:,:,:100,:] & tgt[:,:,:100,:]).sum().float().item()
            total_union_10 += (pred_mask[:,:,:100,:] | tgt[:,:,:100,:]).sum().float().item()



            fn +=(~pred_mask & tgt).sum().float().item() # wrong background prediction
            fp +=(pred_mask & ~tgt).sum().float().item() # wrong foreground prediction
            tp = total_intersect
            tn += (~pred_mask & ~tgt).sum().float().item()

            acc = (tp + tn) / (tp + tn + fp + fn+esp)
            precision = tp / (tp + fp+esp)
            recall = tp / (tp + fn+esp)
            iou = total_intersect / (total_union +esp)
            iou_5 = total_intersect_5 / (total_union_5 + esp)
            iou_10 = total_intersect_10 / (total_union_10 + esp)

            # print(
            #     f"{exp_name}--- acc: {acc} --- recall: {recall} --- precision: {precision} --- iou: {iou} --- iou_5: {iou_5} --- iou_10: {iou_10}")


            # for i in range(seg_bev_pred.shape[0]):
            #     show_img = cv2.imread(img_name[i], -1)
            #     pred_np = seg_bev_pred[i].cpu() > 0.0
            #     pred_np = (pred_np.numpy() * 255).astype(np.uint8)
            #     gt_np = tgt[i].cpu().numpy()
            #     gt_np = (gt_np * 255).astype(np.uint8)
            #     tem_pred = np.zeros((3, pred_np.shape[1], pred_np.shape[2]), np.uint8)
            #     tem_gt = np.zeros((3, pred_np.shape[1], pred_np.shape[2]), np.uint8)
            #
            #     tem_gt += gt_np
            #     tem_pred += pred_np
            #
            #     tem_gt = np.moveaxis(tem_gt, 0, -1)
            #     tem_pred = np.moveaxis(tem_pred, 0, -1)
            #
            #     # print(tem_gt.shape)
            #     # print(tem_pred.shape)
            #     # print(show_img.shape)
            #
            #     #
            #     bev_view = np.concatenate((tem_gt,tem_pred), axis=1)
            #     # # print(bev_view.shape)
            #     ratio = show_img.shape[0] / bev_view.shape[0]
            #     bev_view = cv2.resize(bev_view, (int(ratio*bev_view.shape[1]),show_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            #     #
            #     # # print(bev_view.shape)
            #     # bev_view = cv2.copyMakeBorder(bev_view, 48, 48, 248, 248, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            #     # # print(bev_view.shape, show_img.shape)
            #     bev_view = cv2.flip(bev_view, 0)
            #     #
            #     show_img = np.concatenate([show_img, bev_view], axis=1)
            #     # # show_img = np.moveaxis(show_img, 0, -1)
            #     #
            #     cv2.imwrite( f'{vis_folder}/{img_name[i].split("/")[-1]}',
            #                 show_img)

    print(f"{exp_name}--- acc: {acc} --- recall: {recall} --- precision: {precision} --- iou: {iou} --- iou_5: {iou_5} --- iou_10: {iou_10}")










if __name__ == "__main__":
    from glob import glob
    yaml_files = glob(f"/home/jing/Downloads/Radar_Camera_Fusion/cfgs_trained/*.yml")
    for files in yaml_files:
        exp_name = files.split("/")[-1].replace(".yml",'')
        cfgs = read_cfg(name=files)
        cfgs = Namespace(**cfgs)
        visualization(cfgs,exp_name)