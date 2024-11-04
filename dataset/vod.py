import torchvision
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
import logging
import torchvision.transforms as T
import cv2


class VodData(Dataset):
    def __init__(self, vod_path, radar_folder, grid_conf, is_train, data_aug_conf, useRadar, useCamera):
        super(VodData).__init__()
        self.vox_x = grid_conf["xbound"][-1]
        self.vox_z = grid_conf["zbound"][-1]
        self.is_train = is_train
        self.vod_path = vod_path
        self.useRadar = useRadar
        self.useCamera = useCamera
        self.grid_conf = grid_conf
        self.radar_folder = radar_folder
        self.data_aug_conf = data_aug_conf

        self._setup_directories()
        self._setup_grid_parameters()
        self.frame_numbers = self.get_frame_numbers()


    def get_frame_numbers(self):
        '''
        Get all frame numbers from the val or train txt file
        '''
        frame_num_txt_path = os.path.join(
            self.vod_path, f"{self.radar_folder}/ImageSets/{'train' if self.is_train else 'val'}.txt"
        )
        with open(frame_num_txt_path, 'r') as f:
            return [x.strip() for x in f.readlines()]


    def _setup_directories(self):
        self.radar_calib_dir = os.path.join(self.vod_path, f"{self.radar_folder}/training/calib")
        self.cam_data_dir = os.path.join(self.vod_path, f"{self.radar_folder}/training/image_2")
        self.radar_data_dir = os.path.join(self.vod_path, f"{self.radar_folder}/training/velodyne")
        self.ann_data_dir = os.path.join(self.vod_path, f"{self.radar_folder}/training/label_2")


    def _setup_grid_parameters(self):
        '''
        x: left/ right y: up /down z: backward / forward
        nx: The bev grid number in each dim
        dx: The resolution of each grid
        bev_offset: used to convert the points from camera coordinate to BEV image
        bev_scale: The resolution of each grid (ONLY ON BEV IMAGE)
        '''
        self.nx = np.array([(row[1] - row[0]) / row[2] for row in [
            self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound']
        ]]).astype(np.int32)
        self.dx = np.array([row[2] for row in [
            self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound']
        ]])
        self.bx = np.array([row[0] + row[2] / 2.0 for row in [
            self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound']
        ]])
        self.bev_offset = np.array([self.bx[0] - self.dx[0] / 2, self.bx[2] - self.dx[2] / 2])
        self.bev_scale = np.array([self.dx[0], self.dx[2]])
        # The x z left values are used to make the ROI objects
        # have the coordinate as BEV coordinate (always start from 0)
        # We don't need y value because we use BEV (without the yaxis)
        self.x_left = self.grid_conf['xbound'][0]
        self.z_left = self.grid_conf['zbound'][0]


    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if 'resize_lim' in self.data_aug_conf and self.data_aug_conf['resize_lim'] is not None:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            else:
                resize = self.data_aug_conf['resize_scale']

            resize_dims = (int(fW * resize), int(fH * resize))

            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            crop_offset = self.data_aug_conf['crop_offset']
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:  # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop


    def get_sensor_transforms(self, sensor, frame_number):
        if sensor == 'radar':
            calibration_file = os.path.join(self.radar_calib_dir, f'{frame_number}.txt')
        elif sensor == 'lidar':
            calibration_file = os.path.join(self.lidar_calib_dir, f'{frame_number}.txt')
        else:
            raise AttributeError('Not a valid sensor')

        try:
            with open(calibration_file, "r") as f:
                lines = f.readlines()
                intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
                extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
                extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)
            return intrinsic, extrinsic
        except FileNotFoundError:
            logging.error(f"{frame_number}.txt does not exist at location: {calibration_file}!")
            return None, None

    def get_image_data(self, frame_num):
        img_name = os.path.join(self.cam_data_dir, f"{frame_num}.jpg")
        img = Image.open(img_name)
        intrinsic, extrinsic = self.get_sensor_transforms('radar', frame_num)
        intrinsic_rot = intrinsic[:3, :3]
        W, H = img.size

        resize_dims, crop = self.sample_augmentation()
        sx = resize_dims[0] / float(W)
        sy = resize_dims[1] / float(H)

        intrinsic_rot[0, :] *= sx
        intrinsic_rot[1, :] *= sy

        intrinsic_rot[0, 2] -= crop[0]
        intrinsic_rot[1, 2] -= crop[1]

        img = img.resize(resize_dims, Image.NEAREST)
        img = img.crop(crop)

        normalize_img = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = normalize_img(img)

        return img, extrinsic, intrinsic_rot, img_name


    @staticmethod
    def get_info_from_annotation(ann_data_dir, frame_num):
        annotation_file = os.path.join(ann_data_dir, f"{frame_num}.txt")
        with open(annotation_file, 'r') as f:
            return f.readlines()

    @staticmethod
    def get_rotated_box_points(rot, points, center):
        x, y = points
        xc, yc = center
        tem_x, tem_y = x - xc, y - yc
        rot_x = tem_x * np.cos(rot) - tem_y * np.sin(rot)
        rot_y = tem_x * np.sin(rot) + tem_y * np.cos(rot)
        return rot_x + xc, rot_y + yc


    def generate_center_offset_bev(self, X, Z, center_tensor, vox_x, vox_z, radius=3):
        '''

        :param X: BEV map X length
        :param Z: BEV map Z length
        :param center_tensor: Nx2 camera coordinate -> BVE coordinate
        :param radius: 3
        :return:
        '''

        if center_tensor.nelement() == 0:
            # When nothing in the detection area
            off = torch.zeros((1,2,Z,X))
            mask = torch.zeros((1,Z,X))
            return off, mask
        else:
            N = center_tensor.shape[0]

            center_tensor[:, 0] -= self.x_left
            center_tensor[:, 1] -= self.z_left
            xz = torch.stack([center_tensor[:, 0] / vox_x, center_tensor[:, 1] / vox_z], dim=1)

            grid_z = torch.linspace(0.0, Z - 1, Z)
            grid_z = torch.reshape(grid_z, [Z, 1])
            grid_z = grid_z.repeat(1, X)

            grid_x = torch.linspace(0.0, X - 1, X)
            grid_x = torch.reshape(grid_x, [1, X])
            grid_x = grid_x.repeat(Z, 1)

            grid = torch.stack([grid_x, grid_z], dim=0)  # 2 x Z x X
            xz = xz.reshape(N, 2, 1, 1)
            xz = xz.round()
            grid = grid.reshape(1, 2, Z, X)

            off = grid - xz  # N 2 Z X
            dist_grid = torch.sum(off ** 2, dim=1, keepdim=False)  # N Z X
            mask = torch.exp(-dist_grid / (2 * radius * radius))
            # zero out near zero
            mask[mask < 0.001] = 0.0

            return off, mask


    def generate_seg_bev(self, seg_bev, loc, dim, rotation_y, bev_center, index):

        bev_top_left = (float(loc[0]) + float(dim[1]) / 2, float(loc[2]) + float(dim[2]) / 2)
        bev_bot_right = (float(loc[0]) - float(dim[1]) / 2, float(loc[2]) - float(dim[2]) / 2)
        bev_top_right = (float(loc[0]) - float(dim[1]) / 2, float(loc[2]) + float(dim[2]) / 2)
        bev_bot_left = (float(loc[0]) + float(dim[1]) / 2, float(loc[2]) - float(dim[2]) / 2)

        bev_top_left = self.get_rotated_box_points(-(np.pi / 2 + rotation_y), bev_top_left, bev_center)
        bev_bot_right = self.get_rotated_box_points(-(np.pi / 2 + rotation_y), bev_bot_right, bev_center)
        bev_top_right = self.get_rotated_box_points(-(np.pi / 2 + rotation_y), bev_top_right, bev_center)
        bev_bot_left = self.get_rotated_box_points(-(np.pi / 2 + rotation_y), bev_bot_left, bev_center)

        bev_top_left = np.round((bev_top_left - self.bev_offset) / self.bev_scale).astype(np.int32)
        bev_bot_right = np.round((bev_bot_right - self.bev_offset) / self.bev_scale).astype(np.int32)
        bev_top_right = np.round((bev_top_right - self.bev_offset) / self.bev_scale).astype(np.int32)
        bev_bot_left = np.round((bev_bot_left - self.bev_offset) / self.bev_scale).astype(np.int32)

        pts = np.array([bev_top_left, bev_top_right, bev_bot_right, bev_bot_left], dtype=np.int32)
        seg_bev = cv2.fillPoly(seg_bev, pts=[pts], color=index)
        return seg_bev


    def get_bev_data(self, frame_num):
        seg_bev = np.zeros((self.nx[2], self.nx[0]))
        x_list = []
        z_list = []
        lines = self.get_info_from_annotation(self.ann_data_dir, frame_num)
        for index, line in enumerate(lines):
            line_list = line.strip().split(' ')
            class_name, dim, loc, rotation_y = line_list[0], line_list[-8:-5], line_list[-5:-2], float(line_list[-2])
            bev_center = (float(loc[0]), float(loc[2]))  # x:left/right z:back/front
            x_list.append(bev_center[0])
            z_list.append(bev_center[1])
            seg_bev = self.generate_seg_bev(seg_bev, loc, dim, rotation_y, bev_center, index + 1)

        seg_bev = torch.tensor(seg_bev)

        center_tensor = torch.stack([torch.tensor(x_list), torch.tensor(z_list)], dim=1)
        offset, center = self.generate_center_offset_bev(self.nx[0], self.nx[2], center_tensor, self.vox_x, self.vox_z,
                                                         radius=4)  # N 2 Z X / N Z X

        mask_list = torch.zeros((offset.shape[0], 1, offset.shape[2], offset.shape[3]), dtype=torch.float32)
        for n in range(offset.shape[0]):
            mask_list[n, 0, :, :] = (seg_bev == (n + 1)).float()


        offset *= mask_list
        # print(offset[:,1,:,:].unsqueeze(dim=1)[mask_list>0])
        # torchvision.utils.save_image(offset[0, 0], "/home/jing/Downloads/BevFusion_CR/vis/offset_00.png")

        offset = torch.sum(offset, dim=0)
        center = torch.max(center, dim=0)[0]
        seg_bev = (seg_bev > 0).float()

        return seg_bev.unsqueeze(0), offset, center.unsqueeze(0)


    @staticmethod
    def create_bev_grid(x_range, y_range, z_range):
        """Create a bird's eye view grid based on the given ranges."""
        return np.zeros(
            (
                int((z_range[1] - z_range[0]) / z_range[2]),
                int((x_range[1] - x_range[0]) / x_range[2]),
                int((y_range[1] - y_range[0]) / y_range[2])
            ), dtype=np.float32
        )

    @staticmethod
    def voxelize_pointcloud(num_features, xbound, ybound, zbound, points):
        """Convert a point cloud into a voxel grid."""
        bev_grid_features = [VodData.create_bev_grid(xbound, ybound, zbound) for _ in range(num_features - 3)]

        index = ((points[:, :3] - [xbound[0], ybound[0], zbound[0]]) / [xbound[-1], ybound[-1], zbound[-1]]).astype(int)
        kept = (
                (index[:, 0] >= 0) & (index[:, 0] < bev_grid_features[0].shape[1]) &
                (index[:, 1] >= 0) & (index[:, 1] < bev_grid_features[0].shape[2]) &
                (index[:, 2] >= 0) & (index[:, 2] < bev_grid_features[0].shape[0])
        )

        index = index[kept]
        points = points[kept]

        average_grid_features = []

        for bev_num in range(num_features - 3):
            np.add.at(bev_grid_features[bev_num], (index[:, 2], index[:, 0], index[:, 1]), points[:, 3 + bev_num])

            count_grid = np.zeros_like(bev_grid_features[bev_num])
            np.add.at(count_grid, (index[:, 2], index[:, 0], index[:, 1]), 1)
            count_grid[count_grid == 0] = 1

            average_grid = bev_grid_features[bev_num] / count_grid
            average_grid_features.append(average_grid)

        return np.concatenate(average_grid_features, axis=2)

    def get_radar_data(self, frame_num):
        radar_name = os.path.join(self.radar_data_dir, f"{frame_num}.bin")
        _, extrinsic = self.get_sensor_transforms('radar', frame_num)
        scan = np.fromfile(radar_name, dtype=np.float32).reshape(-1, 7)
        points, features = scan[:, :3], scan[:, 3:6]
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        # points = np.dot(extrinsic, points.T).T
        points = extrinsic.dot(points.T).T
        points = points[:, :3] / points[:, 3].reshape(-1, 1)

        input_scan = np.concatenate([points, features], axis=1)
        radar_features = VodData.voxelize_pointcloud(
            input_scan.shape[1], self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'],
            input_scan
        )
        return radar_features


    def __getitem__(self, index):
        frame_number = self.frame_numbers[index]
        out_dict = {"image": 0,
                    "radar_features": 0,
                    "extrinsic": 0,
                    "intrinsic": 0,
                    "seg_bev": 0,
                    "offset": 0,
                    "center": 0,
                    "post_trans": 0,
                    "post_rot": 0,
                    "img_name": 0}
        if self.useRadar:
            out_dict['radar_features'] = self.get_radar_data(frame_number)
        if self.useCamera:
            img, extrinsic, intrinsic_rot, img_name = self.get_image_data(frame_number)
            out_dict['image'] = img
            out_dict['extrinsic'] = extrinsic
            out_dict['intrinsic'] = intrinsic_rot
            out_dict['img_name'] = img_name

        # seg_bev,offset,center

        out_dict['seg_bev'], out_dict['offset'], out_dict['center'] = self.get_bev_data(frame_number)

        return out_dict

    def __len__(self):
        return len(self.frame_numbers)

def dataloaders(path, grid, nworkers, batch_size, data_aug_conf,
                useRadar=True, useCamera=True):
    # vod_path,radar_folder,grid_conf,is_train,data_aug_conf,vox_x,vox_z,useRadar,useCamera
    train_data = VodData(vod_path=path, radar_folder="radar", grid_conf=grid, is_train=True,
                         data_aug_conf=data_aug_conf, useRadar=useRadar, useCamera=useCamera)
    val_data = VodData(vod_path=path, radar_folder="radar", grid_conf=grid, is_train=False, data_aug_conf=data_aug_conf,
                       useRadar=useRadar, useCamera=useCamera)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=nworkers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=nworkers
    )

    return train_loader, val_loader


if __name__ == "__main__":
    path = "/home/jing/Downloads/view_of_delft_PUBLIC/"
    device = "cuda:0"

    xbound = [-8.0, 8.0, 0.1]
    ybound = [-10.0, 10.0, 20.0]
    zbound = [0, 32, 0.1]
    grid = grid = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound
    }
    resize_lim = [0.8, 1.2]
    res_scale = 0.5
    final_dim = (int(512 * res_scale), int(1024 * res_scale))
    print('resolution:', final_dim)
    crop_offset = int(final_dim[0] * (1 - resize_lim[0]))
    data_aug_conf = {
        'crop_offset': crop_offset,
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'H': 1216, 'W': 1936,
    }
    workers = 1
    batch_size = 1
    use_radar = False
    use_Camera = True

    val_data = VodData(vod_path=path, radar_folder="radar", grid_conf=grid, is_train=False, data_aug_conf=data_aug_conf,
                       useRadar=use_radar, useCamera=use_Camera)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=workers,
    )

    # val_data.get_bev_data("00000")

    iter_val = iter(val_loader)
    for index,sample in enumerate(iter_val):
        img = sample['image'].to(device)
        intrinsic = sample['intrinsic'].to(device)
        seg_bev = sample['seg_bev'].to(device)
        offset = sample['offset'].to(device)
        center = sample['center'].to(device)
        print(seg_bev.shape)

        torchvision.utils.save_image(img[0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/img.png")
        torchvision.utils.save_image(seg_bev[0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/seg.png")
        torchvision.utils.save_image(offset[0, 0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/offset_0.png")
        torchvision.utils.save_image(offset[0, 1], "/home/jing/Downloads/Radar_Camera_Fusion/vis/offset_1.png")
        torchvision.utils.save_image(center[0], "/home/jing/Downloads/Radar_Camera_Fusion/vis/center.png")
        if index == 200:
            break


