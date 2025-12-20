import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 
import pypose as pp

torch.classes.load_library("modules/gaussian_search/build/libgs_search.so")
gs_search_global = torch.classes.gs_search.GS_Search()   


# 工具函数
class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply

# 点云体素下采样
def voxel_down_sample_torch(points: torch.tensor, voxel_size: float):
    _quantization = 1000 # if change to 1, then it would be random sample

    offset = torch.floor(points.min(dim=0)[0]/voxel_size).long()
    grid = torch.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = ((points - center) ** 2).sum(dim=1)**0.5
    dist = dist / dist.max() * (_quantization - 1) # for speed up

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    
    offset = 10**len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset
    idx = torch.empty(unique.shape, dtype=inverse.dtype,
                    device=inverse.device).scatter_reduce_(dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False)
    idx = idx % offset
    return idx

class _gs_searching_backward(Function):  
    @staticmethod
    @custom_fwd
    def forward(ctx, query_points, positions,  rotations, scalings, neighb_vector_grad, neiber_idx, inside_mask):    
        gs_scales =  torch.exp(scalings)
        gs_center = positions[inside_mask]
        gs_rotvec = rotations[inside_mask]
        gs_scale =gs_scales[inside_mask]
        
        d_query_points = torch.zeros_like(query_points, dtype = gs_center.dtype)
        d_gs_center = torch.zeros_like(gs_center, dtype = gs_center.dtype)
        d_gs_rotvec = torch.zeros_like(gs_rotvec, dtype = gs_rotvec.dtype)
        d_gs_scale = torch.zeros_like(gs_scale, dtype = gs_scale.dtype)
        
        neiber_idx = neiber_idx.contiguous()
        
        gs_search_global.compute_grad(query_points, gs_center, gs_rotvec, gs_scale, neighb_vector_grad, neiber_idx, d_query_points, d_gs_center, d_gs_rotvec, d_gs_scale)
        ctx.save_for_backward(positions,  rotations, gs_scales, neighb_vector_grad, neiber_idx, inside_mask)
        
        d_positions = torch.zeros_like(positions)
        d_rotations =  torch.zeros_like(rotations)     # 对旋转向量的导数
        d_scalings = torch.zeros_like(scalings)
        d_positions[inside_mask] = d_gs_center
        d_rotations[inside_mask] = d_gs_rotvec
        d_scalings[inside_mask] =d_gs_scale
        
        return d_query_points, d_positions, d_rotations, d_scalings
    
    @staticmethod
    @custom_bwd
    def backward(ctx, d_query_points_grad, d_gs_center_grad, d_gs_rotvec_grad, d_gs_scale_grad):
        positions,  rotations, gs_scales, neighb_vector_grad, neiber_idx, inside_mask = ctx.saved_tensors    # 这里的gs_scale已经经过了exp函数
        gs_center = positions[inside_mask]
        gs_rotvec = rotations[inside_mask]
        gs_scale = gs_scales[inside_mask]
        
        gs_rotvec_grad = torch.zeros_like(gs_rotvec, dtype = gs_rotvec.dtype)
        gs_scale_grad = torch.zeros_like(gs_scale, dtype = gs_scale.dtype)
        neighb_vector_grad_grad = torch.zeros_like(neighb_vector_grad, dtype = gs_center.dtype)
        
        gs_search_global.compute_grad_grad(d_query_points_grad, gs_rotvec, gs_scale, neighb_vector_grad, neiber_idx, gs_rotvec_grad, gs_scale_grad, neighb_vector_grad_grad)

        d_orientations =  torch.zeros_like(rotations)     # 对旋转向量的导数
        d_scales = torch.zeros_like(gs_scales)
        d_orientations[inside_mask] = gs_rotvec_grad
        d_scales[inside_mask] =gs_scale_grad
        
        return None, None, d_orientations, d_scales, neighb_vector_grad_grad, None, None

# 自己添加
class _gs_searching(Function):    
    @staticmethod
    @custom_fwd
    def forward(ctx, query_points, positions, rotations, scalings, query_nn_k, resolution, gss_vox_size):    # orientations为旋转向量(x,y,z)
        bounding_min, _ = torch.min(query_points, dim=0)
        bounding_max, _ = torch.max(query_points, dim=0)
        bounding_min = bounding_min - 3*resolution      # bounding边界膨胀
        bounding_max = bounding_max + 3*resolution
        gs_scales = torch.exp(scalings)    # 相当于给scale加了一个激活函数，主要是防止小于等于0
        inside_mask = gs_search_global.find_gs_by_boundingbox(positions, rotations, gs_scales, bounding_min, bounding_max)
        gs_center = positions[inside_mask]
        gs_rotvec = rotations[inside_mask]
        gs_scale =  gs_scales[inside_mask]
        
        query_points = query_points.contiguous()
        neiber_dis2, neiber_idx = gs_search_global.find_neibors(query_points, gs_center, gs_rotvec, gs_scale, bounding_min, bounding_max, gss_vox_size, 32)
        
        neiber_dis2[neiber_idx==-1] = 9e3 
        sorted_dist2, sorted_neigh_idx = torch.sort(neiber_dis2, dim=1) 
        sorted_idx = neiber_idx.gather(1, sorted_neigh_idx)
        sorted_dist2 = sorted_dist2[:,:query_nn_k] 
        neiber_idx = sorted_idx[:,:query_nn_k] 
        
        gs_center_neiber = gs_center[neiber_idx]
        gs_rotvec_neiber = gs_rotvec[neiber_idx]
        gs_scale_neiber = gs_scale[neiber_idx]

        neighb_vector = query_points.view(-1, 1, 3) -  gs_center_neiber
        rots = gs_rotvec_neiber.reshape(-1, 3)
        rots = pp.so3(rots).lview(neiber_idx.shape[0], neiber_idx.shape[1]) 
        rots = rots.Inv().Exp()
        neighb_vector = rots @ neighb_vector 
        neighb_vector = neighb_vector / gs_scale_neiber

        ctx.save_for_backward(query_points, positions, rotations, scalings, neiber_idx, inside_mask)
        ctx.dims = [positions.shape[0]]
        
        return neighb_vector, neiber_idx, inside_mask
    
    @staticmethod
    @custom_bwd
    def backward(ctx, neighb_vector_grad, neiber_idx_grad, inside_mask_grad):      
        query_points, positions, rotations, scalings, neiber_idx, inside_mask = ctx.saved_tensors 
        
        d_query_points, d_positions, d_rotations, d_scaling = _gs_searching_backward.apply(query_points, positions,  rotations, scalings, neighb_vector_grad, neiber_idx, inside_mask)
        
        return d_query_points, d_positions, d_rotations, d_scaling, None, None, None    

gs_searching_module = _gs_searching.apply

class LocalSDF(nn.Module):

    def __init__(self, resolution, query_nn_k, gss_vox_size) -> None:
        super().__init__()
        
        self.resolution = resolution 
        
        self.updata_mask  = None
        self.neiber_idx = None

        self.device = "cuda"
        self.dtype = torch.float32
        
        self.positions = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        self.rotations = torch.empty((0,3), dtype=self.dtype, device=self.device)
        self.scalings = torch.empty((0,3), dtype=self.dtype, device=self.device)
        
        self.pts_accum = None
        self.normals_accum = None
        
        # 自己添加
        self.query_nn_k: int = query_nn_k
        self.weighted_first = False 
        self.gss_vox_size = gss_vox_size
        self.to(self.device)

    def is_empty(self):
        return self.positions.shape[0] == 0
    
    def init_from_preload_params(self, params):     
        neup_params = params["local_sdfs.positions"]
        n_neup_params = neup_params.shape[0]
        self.positions = torch.nn.Parameter(torch.empty((n_neup_params,3), dtype=self.dtype, device=self.device))    
        self.rotations = torch.nn.Parameter(torch.empty((n_neup_params,3), dtype=self.dtype, device=self.device))
        self.scalings = torch.nn.Parameter(torch.empty((n_neup_params,3), dtype=self.dtype, device=self.device))           
        self.xyz_gradient_accum = torch.zeros(self.positions.shape[0], device=self.device)
        self.denom = torch.zeros(self.positions.shape[0], device=self.device)
    
    def update_in_svh(self, global_pcd):    
        global_points = torch.tensor(np.asarray(global_pcd.points), device = self.device).float()
        global_normals = torch.tensor(np.asarray(global_pcd.normals), device = self.device).float()  # 法向
        nls_norm = torch.norm(global_normals, dim = -1, keepdim=True)
        global_normals = global_normals/(nls_norm+1e-5)
        
        cur_resolution = self.resolution
        sample_idx = voxel_down_sample_torch(global_points, cur_resolution) 
        self.positions = global_points[sample_idx]
        
        neural_normals = global_normals[sample_idx]
        # 根据点云的法向生成旋转矩阵
        normal_t = torch.tensor([0.0,0.0,1.0], device=self.device).reshape(1,3)
        angle = torch.arccos(torch.sum(normal_t*neural_normals, dim = -1))   
        axis = torch.cross(normal_t, neural_normals, dim = -1)    # [n_points, 3]
        norm = torch.norm(axis, dim = -1)
        mask = (norm>1e-5)
        axis[mask] = axis[mask]/norm[mask][:, None]   
        
        # 利用pypose计算
        rot_vector = torch.zeros(self.positions.shape[0], 3, device=self.device)
        rot_vector[mask] = axis[mask] * angle[mask][:, None]
        self.rotations = rot_vector    
        self.scalings = torch.ones(self.positions.shape[0], 3, device=self.device) * math.log(self.resolution)   # 相当于半径
        
        # 与致密化相关的变量
        self.xyz_gradient_accum = torch.zeros(self.positions.shape[0], device=self.device)
        self.denom = torch.zeros(self.positions.shape[0], device=self.device)
        
    # 数据加载完及样本生成后，为优化作准备
    def prepare_for_optimization(self):
        self.positions = nn.Parameter(self.positions)    
        self.rotations = nn.Parameter(self.rotations)        
        self.scalings = nn.Parameter(self.scalings)                               
    
    def query_feature(self, query_points: torch.Tensor): 
        neighb_vector, neiber_idx, inside_mask = gs_searching_module(query_points, self.positions, self.rotations, self.scalings, self.query_nn_k, self.resolution, self.gss_vox_size)
        self.updata_mask = inside_mask
        self.neiber_idx = neiber_idx
        nn_counts = (neiber_idx >= 0).sum(dim=-1) 
        valid_mask = neiber_idx >= 0 # [N, K]

        geo_features_vector = neighb_vector   

        eps = 1e-15 
        dists2 = (neighb_vector*neighb_vector).sum(dim = -1)     
        weight_vector = trunc_exp(-0.5*dists2)     
        
        weight_vector[~valid_mask] = 0. 
        weight_vector[nn_counts == 0] = eps 
        
        weight_row_sums = torch.sum(weight_vector, dim=1).unsqueeze(1)
        weight_vector = torch.div(weight_vector, weight_row_sums)
        weight_vector[~valid_mask] = 0. 
        weight_vector = weight_vector.unsqueeze(-1) 

        if self.weighted_first:    
            geo_features_vector = torch.sum(geo_features_vector * weight_vector, dim=1) 

        weight_vector = weight_vector.squeeze()
        # weight_vector = weight_vector.detach()   # 不对权重求导，效果极差
        
        return geo_features_vector, weight_vector, nn_counts
    
    # 添加致密化状态
    @torch.no_grad()
    def add_densification_stats(self):
        gs_position_grad = self.positions.grad[self.updata_mask]
        gs_position_grad_norm = torch.norm(gs_position_grad, dim = -1)   # 得到gs位置的梯度模长
        rot_vec = self.rotations[self.updata_mask]
        so3 = pp.so3(rot_vec)
        gs_normal = so3.matrix()[:,2].reshape(-1, 3)   # 从旋转矩阵中得到法向
        grad_on_normal = torch.abs((gs_position_grad*gs_normal).sum(-1))
        grad_on_xy = torch.sqrt(gs_position_grad_norm*gs_position_grad_norm-grad_on_normal*grad_on_normal)
        
        self.xyz_gradient_accum[self.updata_mask] += grad_on_xy
        self.denom[self.updata_mask] += 1
    
    
    def densification_postfix(self, new_positions, new_rotations, new_scalings, optimizer):
        tensors_dict = {
            "positions": new_positions,
            "rotations": new_rotations,
            "scalings": new_scalings
        }
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] == "sdf_net":
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        
        self.positions = optimizable_tensors["positions"]
        self.rotations = optimizable_tensors["rotations"]
        self.scalings = optimizable_tensors["scalings"]
        self.xyz_gradient_accum = torch.zeros(self.positions.shape[0], device=self.device)
        self.denom = torch.zeros(self.positions.shape[0], device=self.device)
    
    def prune_optimizer(self, prune_filter, optimizer):
        valid_points_mask = ~prune_filter
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] == "sdf_net":
                continue
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_points_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_points_mask]
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][valid_points_mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][valid_points_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
                
        self.positions = optimizable_tensors["positions"]
        self.rotations = optimizable_tensors["rotations"]
        self.scalings = optimizable_tensors["scalings"]
        self.xyz_gradient_accum = torch.zeros(self.positions.shape[0], device=self.device)
        self.denom = torch.zeros(self.positions.shape[0], device=self.device)
    
    # 致密化以及裁剪
    @torch.no_grad()
    def densify_and_prune(self, cfg, iter, optimizer, neural_map):
        grad_threshold=cfg["LocalSDF"]["grad_threshold_densitify"]
        scale_threshold= 1.2*self.resolution # 0.8*self.resolution
        sdf_threshold=cfg["LocalSDF"]["sdf_threshold_densitify"]
        if iter >cfg["Train"]["sdf_th_after_iters"]:   # N次迭代后增加致密化sdf阈值
            sdf_threshold=cfg["LocalSDF"]["sdf_threshold_densitify"] + 0.1
        consider_large_gs = cfg["LocalSDF"]["consider_large_gs"]
        # 函数正式开始
        # 裁剪，删除sdf绝对值大于阈值的局部SDF
        spatial_vox_size = 200
        bbox_min = (torch.min(self.positions, dim=0).values-1).int()
        bbox_max = (torch.max(self.positions, dim=0).values+1).int()
        bbox_len = bbox_max - bbox_min
        positions_axis1 = None
        positions_axis2 = None
        sdf = torch.zeros(self.positions.shape[0]).to(self.device)
        if torch.argmin(bbox_len).item() == 0:
            list_a = torch.arange(bbox_min[1], bbox_max[1], spatial_vox_size)
            list_b = torch.arange(bbox_min[2], bbox_max[2], spatial_vox_size)
            positions_axis1 = self.positions[:, 1]
            positions_axis2 = self.positions[:, 2]
        elif torch.argmin(bbox_len).item() == 1:
            list_a = torch.arange(bbox_min[0], bbox_max[0], spatial_vox_size)
            list_b = torch.arange(bbox_min[2], bbox_max[2], spatial_vox_size)
            positions_axis1 = self.positions[:, 0]
            positions_axis2 = self.positions[:, 2]
        elif torch.argmin(bbox_len).item() == 2:
            list_a = torch.arange(bbox_min[0], bbox_max[0], spatial_vox_size)
            list_b = torch.arange(bbox_min[1], bbox_max[1], spatial_vox_size)
            positions_axis1 = self.positions[:, 0]
            positions_axis2 = self.positions[:, 1]
        grid_a, grid_b = torch.meshgrid(list_a, list_b)
        grid = torch.stack([grid_a, grid_b], dim = -1)
        # print(grid.shape)
        grid = grid.reshape(-1, 2)
        for i in range(grid.shape[0]):
            vox_min = grid[i]
            vox_max = grid[i] + spatial_vox_size
            mask = (positions_axis1 >= vox_min[0]) & (positions_axis1 < vox_max[0]) & (positions_axis2 >= vox_min[1]) & (positions_axis2 < vox_max[1])
            positions = self.positions[mask]
            if len(positions) == 0:
                continue
            sdf_t, _ = neural_map.query_sdf(positions)
            sdf[mask] = sdf_t
                
        if consider_large_gs:
            selected_pts_mask = (torch.abs(sdf) < sdf_threshold) | (self.xyz_gradient_accum == 0) | (torch.max(torch.exp(self.scalings), dim = -1).values > 2 * sdf_threshold)   # 保留sdf值小于阈值的局部SDF，或尺度较大的局部SDF
        else:
            selected_pts_mask = (torch.abs(sdf) < sdf_threshold) | (self.xyz_gradient_accum == 0) 
        selected_pts_mask = torch.abs(sdf) < sdf_threshold
        new_positions = self.positions[selected_pts_mask]
        new_rotations = self.rotations[selected_pts_mask]
        new_scalings = self.scalings[selected_pts_mask]
        grads = self.xyz_gradient_accum[selected_pts_mask] / self.denom[selected_pts_mask] 
        prune_filter = ~selected_pts_mask
        # print("prune num: %d"%prune_filter.sum().item())
        self.prune_optimizer(prune_filter, optimizer)
        
        # grads = self.xyz_gradient_accum / self.denom  
        grads[grads.isnan()] = 0.0

        # 克隆
        gs_scale = torch.exp(self.scalings)
        selected_pts_mask = (grads > grad_threshold) & (torch.max(gs_scale[:, :2], dim = -1).values <= scale_threshold)   # 只判断前两维的scale，因为第三维为法向，法向方向不考虑
        # print("clone num: %d"%selected_pts_mask.sum().item())
        if selected_pts_mask.sum() > 0:
            stds = gs_scale[selected_pts_mask]*1.6
            stds[..., -1] = 0    # 第三维为0，只沿与法向垂直的方向移动
            means =torch.zeros((stds.size(0), 3),device=self.device)
            samples = torch.normal(mean=means, std=stds)
            rot_vec = self.rotations[selected_pts_mask]
            so3 = pp.so3(rot_vec)
            new_positions = so3.Exp()*samples+self.positions[selected_pts_mask]
            # new_positions = self.positions[selected_pts_mask]
            new_rotations = self.rotations[selected_pts_mask]
            new_scalings = self.scalings[selected_pts_mask]
            self.densification_postfix(new_positions, new_rotations, new_scalings, optimizer)

        # 分裂
        gs_scale = torch.exp(self.scalings)
        padded_grad = torch.zeros((self.positions.shape[0]), device=self.device)
        padded_grad[:grads.shape[0]] = grads
        if consider_large_gs:
            selected_pts_mask = ((padded_grad > grad_threshold) & (torch.max(gs_scale[:, :2], dim = -1).values > scale_threshold)) | (torch.max(gs_scale[:, :2], dim = -1).values > 2 * self.resolution)    # 改过的
        else:
            selected_pts_mask = (padded_grad > grad_threshold) & (torch.max(gs_scale[:, :2], dim = -1).values > scale_threshold)     # 原来的
        # print("splite num: %d"%selected_pts_mask.sum().item())
        if selected_pts_mask.sum() > 0:
            N = 2  # 分裂为2个
            stds = gs_scale[selected_pts_mask].repeat(N,1)
            stds[..., -1] = 0     # 第三维为0，只沿与法向垂直的方向移动
            means =torch.zeros((stds.size(0), 3),device=self.device)
            samples = torch.normal(mean=means, std=stds)
            rot_vec = self.rotations[selected_pts_mask].repeat(N, 1)
            so3 = pp.so3(rot_vec)
            new_positions = so3.Exp()*samples+self.positions[selected_pts_mask].repeat(N,1)
            new_rotations = self.rotations[selected_pts_mask].repeat(N,1)
            new_scalings = gs_scale[selected_pts_mask].repeat(N,1)
            indices = torch.max(new_scalings[:, :2], dim = -1).indices
            new_scalings[:, indices] /= 1.6      # 在与法向垂直的平面上分裂，对该平面主方向上的尺度进行缩减
            # new_scales[:, :2] /= 1.6
            new_scalings = torch.log(new_scalings)
            # new_scales = torch.log(gs_scale[selected_pts_mask].repeat(N,1) / 1.6)
            self.densification_postfix(new_positions, new_rotations, new_scalings, optimizer)
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
            self.prune_optimizer(prune_filter, optimizer)
            
        torch.cuda.empty_cache()
        
            
