import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.cm as cm

def gradient(inputs, outputs):
    d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(    
        outputs=outputs,  
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


class NeuralMap(nn.Module):
    def __init__(self,
                 local_sdfs,
                 num_layers=2,
                 hidden_dim=64,
    ):
        super().__init__()
        self.local_sdfs = local_sdfs
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = 3 
        sdf_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = hidden_dim
            sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.sdf_net = nn.ModuleList(sdf_net)
    
    def query_sdf(self, x):
        # x: [N, 3]
        x = x.reshape(-1,3)   # x: [B, 3]
        geo_feature, weight_knn, nn_counts = self.local_sdfs.query_feature(x)   
        
        h = geo_feature

        for l in range(self.num_layers):
            h = self.sdf_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sdf = h     

        if not self.local_sdfs.weighted_first:    
            sdf = torch.sum(sdf.squeeze() * weight_knn, dim=1) # N
        
        sdf = sdf.squeeze()
        return sdf, nn_counts
    

    def query_in_basis_sdf(self, x):
        # x: [N, 3]
        x = x.reshape(-1,3)   # x: [B, 3]
        h = x
        for l in range(self.num_layers):
            h = self.sdf_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sdf = h    
        
        sdf = sdf.squeeze()
        return sdf
            
            
    @torch.no_grad()
    def vis_basis_sdf(self, resolution=256, vis = "all"):
        range = 3
        if vis == "all" or vis == "xz":
            # xz平面
            x = torch.linspace(-range, range, resolution)
            z = torch.linspace(-range, range, resolution)
            # print(z)
            zz, xx = torch.meshgrid(z, x)
            yy = torch.zeros_like(xx)
            sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float()
            shape_ = sampled_xyz.shape
            sampled_xyz = sampled_xyz.reshape(-1,3).to("cuda")
            sdf = self.query_in_basis_sdf(sampled_xyz)
            sdf = sdf.reshape(shape_[0], shape_[1])
            sdf = sdf.detach().cpu().numpy()
            
            min_sdf = -1.2
            max_sdf = 1.2
            sdf_show = np.clip((sdf - min_sdf) / (max_sdf - min_sdf), 0., 1.)
            sdf_show = sdf_show.reshape(-1)
            # color_map = cm.get_cmap("bwr") # or 'jet'
            color_map = cm.get_cmap("coolwarm")     
            # color_map = cm.get_cmap("RdYlGn") 
            colors = color_map(sdf_show)[:, :3]
            colors = (colors * 255).astype(np.uint8)
            sdf_map_xz = colors.reshape(shape_[0], shape_[1], 3)
            
            cv2.imshow("Basis_SDF_xz", sdf_map_xz)
            cv2.waitKey(1)
        
        if vis == "all" or vis == "yz":
            # yz平面
            y = torch.linspace(-range, range, resolution)
            z = torch.linspace(range, -range, resolution)
            zz, yy = torch.meshgrid(z, y)
            xx = torch.zeros_like(yy)
            sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float()
            shape_ = sampled_xyz.shape
            sampled_xyz = sampled_xyz.reshape(-1,3).to("cuda")
            sdf = self.query_in_basis_sdf(sampled_xyz)
            sdf = sdf.reshape(shape_[0], shape_[1])
            sdf = sdf.detach().cpu().numpy()
            
            min_sdf = -1.2
            max_sdf = 1.2
            sdf_show = np.clip((sdf - min_sdf) / (max_sdf - min_sdf), 0., 1.)
            sdf_show = sdf_show.reshape(-1)
            # color_map = cm.get_cmap("bwr") # or 'jet'
            color_map = cm.get_cmap("coolwarm") 
            # color_map = cm.get_cmap("RdYlGn") 
            colors = color_map(sdf_show)[:, :3]
            colors = (colors * 255).astype(np.uint8)
            sdf_map_yz = colors.reshape(shape_[0], shape_[1], 3)
            
            cv2.imshow("Basis_SDF_yz", sdf_map_yz)
            cv2.waitKey(1)
            
    
