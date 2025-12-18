import os
from torch.utils.data import Dataset
import torch
import numpy as np
import open3d as o3d

class LiDARDataset(Dataset):
    def __init__(self, 
                pc_dir, 
                calib_file,
                pose_file,
                min_range = 2.5,
                max_range = 20.0,
                sor_nn = 25,
                sor_std = 2.5,
                use_filter = True):
        super().__init__()
        
        self.pc_dir = pc_dir
        self.min_range = min_range
        self.max_range = max_range
        self.sor_nn = sor_nn
        self.sor_std = sor_std
        self.use_filter = use_filter
        
        #加载标定文件
        self.calib = {}
        if calib_file is not None:
            calib_ = open(calib_file)
            for line in calib_:
                key, content = line.strip().split(":")
                values = [float(v) for v in content.strip().split()]
                pose_ = np.zeros((4, 4))
                pose_[0, 0:4] = values[0:4]
                pose_[1, 0:4] = values[4:8]
                pose_[2, 0:4] = values[8:12]
                pose_[3, 3] = 1.0
                self.calib[key] = pose_
            calib_.close()
        else:
            self.calib['Tr'] = np.eye(4)
        
        Tr = self.calib["Tr"]
        Tr_inv = np.linalg.inv(Tr)
        
        #加载位姿
        pose_file = open(pose_file)
        poses = []
        for line in pose_file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            pose = np.array(pose)
            poses.append(
                np.matmul(Tr_inv, np.matmul(pose, Tr))  # lidar pose in world frame
            )
        pose_file.close()
        self.Ts = np.concatenate(poses, axis = 0).reshape(-1, 4, 4)
        
        # point cloud files
        pc_filenames = os.listdir(pc_dir)
        self.pc_filenames = sorted(pc_filenames)
        
        
    def __len__(self):
        return len(self.pc_filenames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        pc_file = self.pc_dir + self.pc_filenames[idx]
        if ".bin" in pc_file:
            pcd_bin = np.fromfile(pc_file, dtype=np.float32).reshape((-1, 4))[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_bin)
        elif ".ply" in pc_file or ".pcd" in pc_file:
            pcd = o3d.io.read_point_cloud(pc_file)
        
        if self.use_filter:
            pcd = pcd.remove_statistical_outlier(
                    self.sor_nn, self.sor_std, print_progress=False
                )[0]
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        if len(normals) == 0 or np.linalg.norm(normals, axis=1).sum()==0:
            normals = None
        
        ray_length = np.linalg.norm(points, axis=1)
        mask = (ray_length> self.min_range) & (ray_length <= self.max_range)
        points = points[mask].astype(np.float32)
        if normals is not None:
            normals = normals[mask].astype(np.float32)
        
        T = self.Ts[idx].astype(np.float32)
        
        return points, normals, T
