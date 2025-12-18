import torch
import torch.nn.functional as F
import numpy as np
import Thirdparty.skimage.measure as skimage_measure
import open3d as o3d
from tqdm import *
import network

def expand_data(batch, data, replace=False):
    cat_fn = np.concatenate
    if torch.is_tensor(data):
        cat_fn = torch.cat
    if batch is None:
        batch = data
    else:
        if replace is False:
            batch = cat_fn((batch, data))
        else:
            batch[-1] = data[0]
    return batch

def transform_3D_grid(grid_3d, transform=None, scale=None):
    if scale is not None:
        grid_3d = grid_3d * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)

        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d

#批量光线与orient包围盒求交，得到最小最大的深度（bounds_extents为orient包围盒的长宽高，inv_bounds_transform为世界坐标系到包围盒坐标系的变换，
# T_WC_batch为光线所在图像的位姿，dirs_C_batch为光线的方向）
def  ray_intersect(bounds_extents, inv_bounds_transform, origins, dirs_W):   
    bounds_extents = bounds_extents/0.9   #稍微扩大点包围盒
    zero_row = torch.zeros_like(bounds_extents).to(bounds_extents.device)
    bounding_box = torch.stack((zero_row, bounds_extents), dim = 0)
    bounding_box = bounding_box-(bounds_extents/2)[None, :]    #得到orient包围盒

    with torch.set_grad_enabled(False):
        origins_trans = transform_3D_grid(origins, transform=inv_bounds_transform)
        origins_trans = origins_trans.squeeze()
        R_trans = inv_bounds_transform[None, :3, :3]
        dir_trans = (R_trans* dirs_W[..., None, :]).sum(dim=-1)    #以上求的是在orient变换后的图像原点和方向，方便下面与orient包围盒求交   

        tminmax_x = (bounding_box[:,0][None, :] - origins_trans[:, 0][:, None]) / dir_trans[:, 0][:, None]
        min_x, _ = tminmax_x.min(dim = 1)
        max_x, _ = tminmax_x.max(dim = 1)
        tminmax_y = (bounding_box[:,1][None, :] - origins_trans[:, 1][:, None]) / dir_trans[:, 1][:, None]
        min_y, _ = tminmax_y.min(dim = 1)
        max_y, _ = tminmax_y.max(dim = 1)
        tminmax_z = (bounding_box[:,2][None, :] - origins_trans[:, 2][:, None])/ dir_trans[:, 2][:, None]
        min_z, _ = tminmax_z.min(dim = 1)
        max_z, _ = tminmax_z.max(dim = 1)

        min, _ = torch.cat((min_x.unsqueeze(1), min_y.unsqueeze(1), min_z.unsqueeze(1)), dim = 1).max(dim = 1)
        max, _ = torch.cat((max_x.unsqueeze(1), max_y.unsqueeze(1), max_z.unsqueeze(1)), dim = 1).min(dim = 1)
        tminmax =  torch.cat((min.unsqueeze(1), max.unsqueeze(1)), dim = 1)
        tminmax[torch.where(min > max)] = 0
        tminmax[torch.where(tminmax < 0)] = 0
        mask = max > min    #mask是为了筛选与包围盒相交的射线
        
    return tminmax, mask


def get_data(scene_dataset, idxs, is_est_normal = True, device = "cuda"):  #得到点云中坐标和位姿
    pc_s = []
    T_s = []
    normal_s = []
    min_len = 999999999
    pcd_fm = o3d.geometry.PointCloud()
    for idx in idxs:
        pc_np, normal_np, T_np = scene_dataset[idx]
        if pc_np.shape[0] < 100:
            continue
        if pc_np.shape[0] < min_len:
            min_len = pc_np.shape[0]

        pcd_fm.points = o3d.utility.Vector3dVector(pc_np)
        if normal_np is None and is_est_normal:
            pcd_fm = pcd_fm.transform(T_np)     #变换到世界坐标系下
            pcd_fm.estimate_normals()    # 估计法向
            normal_np = np.asarray(pcd_fm.normals)
            pc_np = np.asarray(pcd_fm.points)
            rays_o = T_np[:3,3]
            rays_d = rays_o[None, :] - pc_np
            ranges = np.linalg.norm(rays_d, axis=1)
            rays_d = rays_d/(ranges[:, None]+1e-5)
            dd = (normal_np*rays_d).sum(axis = -1)
            mask = dd < 0
            normal_np[mask] =  - normal_np[mask]     # 利用视线纠正法向方向
        elif normal_np is not None and is_est_normal:
            pcd_fm.normals = o3d.utility.Vector3dVector(normal_np)
            pcd_fm = pcd_fm.transform(T_np)     #变换到世界坐标系下
            normal_np = np.asarray(pcd_fm.normals)
            pc_np = np.asarray(pcd_fm.points)
        else:
            pcd_fm = pcd_fm.transform(T_np)     #变换到世界坐标系下
            pc_np = np.asarray(pcd_fm.points)

        pc = torch.from_numpy(pc_np).float()
        T = torch.from_numpy(T_np).float()
        pc_s.append(pc)
        T_s.append(T)
        if normal_np is not None and is_est_normal:
            normal = torch.from_numpy(normal_np).float()
            normal_s.append(normal)
    
    pc_batch = None
    T_batch = None
    normal_batch = None
    for i in range(len(T_s)):
        pc = pc_s[i]
        T = T_s[i]
        
        ind = (torch.rand(min_len) * pc.shape[0]).long()
        pc = pc[ind]
        pc = pc[None, ...]
        T = T[None, ...]
        
        pc_batch = expand_data(pc_batch, pc)
        T_batch = expand_data(T_batch, T)
        
        if len(normal_s)!=0:
            normal = normal_s[i]
            normal = normal[ind]
            normal = normal[None, ...]
            normal_batch = expand_data(normal_batch, normal)
            
    
    pc_batch = pc_batch.to(device)
    T_batch = T_batch.to(device)
    if normal_batch is not None:
        nls_np_norm = torch.norm(normal_batch, dim = -1)
        normal_batch = normal_batch/(nls_np_norm[:, :, None]+1e-5)   # 法向归一化
        normal_batch = normal_batch.to(device)
    
    return pc_batch, normal_batch, T_batch

def load_data_buffer(scene_dataset, is_est_normal = True):  #得到点云中坐标和位姿
    pc_s = []
    T_s = []
    normal_s = []
    min_len = 999999999
    n_points_total = 0
    pcd_fm = o3d.geometry.PointCloud()
    for idx in tqdm(range(len(scene_dataset))):
        pc_np, normal_np, T_np = scene_dataset[idx]
        if pc_np.shape[0] < 100:
            continue
        if pc_np.shape[0] < min_len:
            min_len = pc_np.shape[0]
        n_points_total += pc_np.shape[0]

        pcd_fm.points = o3d.utility.Vector3dVector(pc_np)
        if normal_np is None and is_est_normal:
            pcd_fm = pcd_fm.transform(T_np)     #变换到世界坐标系下
            pcd_fm.estimate_normals()    # 估计法向
            normal_np = np.asarray(pcd_fm.normals)
            pc_np = np.asarray(pcd_fm.points)
            rays_o = T_np[:3,3]
            rays_d = rays_o[None, :] - pc_np
            rays_d[:, 2] += 3 #5      # 把视点增高有助于远离视点的法向的准确性
            ranges = np.linalg.norm(rays_d, axis=1)
            rays_d = rays_d/(ranges[:, None]+1e-5)
            dd = (normal_np*rays_d).sum(axis = -1)
            mask = dd < 0
            normal_np[mask] =  - normal_np[mask]     # 利用视线纠正法向方向
        elif normal_np is not None and is_est_normal:
            pcd_fm.normals = o3d.utility.Vector3dVector(normal_np)
            pcd_fm = pcd_fm.transform(T_np)     #变换到世界坐标系下
            normal_np = np.asarray(pcd_fm.normals)
            pc_np = np.asarray(pcd_fm.points)
        else:
            pcd_fm = pcd_fm.transform(T_np)     #变换到世界坐标系下
            pc_np = np.asarray(pcd_fm.points)

        pc = torch.from_numpy(pc_np).float()
        T = torch.from_numpy(T_np).float()
        pc_s.append(pc)
        T_s.append(T)
        if normal_np is not None and is_est_normal:
            normal = torch.from_numpy(normal_np).float()
            normal_s.append(normal)
    
    n_points_mean = n_points_total//len(pc_s)
    pc_batch = None
    T_batch = None
    normal_batch = None
    for i in tqdm(range(len(T_s))):
        pc = pc_s[i]
        T = T_s[i]
        ind = (torch.rand(n_points_mean) * pc.shape[0]).long()
        pc = pc[ind]
        pc = pc[None, ...]
        T = T[None, ...]
        
        pc_batch = expand_data(pc_batch, pc)
        T_batch = expand_data(T_batch, T)
        
        if len(normal_s)!=0:
            normal = normal_s[i]
            normal = normal[ind]
            normal = normal[None, ...]
            normal_batch = expand_data(normal_batch, normal)
            
    if normal_batch is not None:
        nls_np_norm = torch.norm(normal_batch, dim = -1)
        normal_batch = normal_batch/(nls_np_norm[:, :, None]+1e-5)   # 法向归一化
        normal_batch = normal_batch
    
    return pc_batch, normal_batch, T_batch

##分配局部SDF，以及向哈希表中分配体素，无返回值
@torch.no_grad()
def allocate_localsdfs_in_svh(svh, scene_dataset, idxs, local_sdfs, res_scale = 2, down_voxel_size = 0.02):  
    pcd_np = None
    nls_np = None
    pcd_fm = o3d.geometry.PointCloud()

    print("load point cloud......")
    n_total = 0
    for idx in tqdm(idxs):
        pc_np, normal_np, T_np = scene_dataset[idx]
        n_total += pc_np.shape[0]//res_scale
    
    n_count = 0
    pcd_np = np.empty((n_total, 3), dtype=np.float32)
    nls_np = np.empty((n_total, 3), dtype=np.float32)
    for idx in tqdm(idxs):
        pc_np, normal_np, T_np = scene_dataset[idx]
        id = np.random.randint(0, pc_np.shape[0], pc_np.shape[0]//res_scale)    #筛选点云
        pc_np = pc_np[id]
        # 点云累积，用于分配哈希网格
        pcd_fm.points = o3d.utility.Vector3dVector(pc_np)
        if normal_np is not None:     # 如果原始点云有法向，则利用原始法向
            normal_np = normal_np[id]
            pcd_fm.normals = o3d.utility.Vector3dVector(normal_np)
            pcd_fm = pcd_fm.transform(T_np)     #变换到世界坐标系下
            normal_np = np.asarray(pcd_fm.normals, dtype=np.float32)
            pc_np = np.asarray(pcd_fm.points, dtype=np.float32)
        else:       # 不然就估计法向，然后用视线方向纠正
            pcd_fm = pcd_fm.transform(T_np)     #变换到世界坐标系下
            pcd_fm.estimate_normals()    # 估计法向
            normal_np = np.asarray(pcd_fm.normals, dtype=np.float32)
            pc_np = np.asarray(pcd_fm.points, dtype=np.float32)
            rays_o = T_np[:3,3]
            rays_d = rays_o[None, :] - pc_np
            rays_d[:, 2] += 3 #5      # 把视点增高有助于远离视点的法向的准确性
            ranges = np.linalg.norm(rays_d, axis=1)
            rays_d = rays_d/(ranges[:, None]+1e-5)
            dd = (normal_np*rays_d).sum(axis = -1)
            mask = dd < 0
            normal_np[mask] =  - normal_np[mask]     # 利用视线纠正法向方向

        nn = len(pc_np)
        pcd_np[n_count:n_count+nn] = pc_np
        nls_np[n_count:n_count+nn] = normal_np
        n_count += nn
            
    # 归一化法向      
    nls_np_norm = np.linalg.norm(nls_np, axis = -1)
    nls_np = nls_np/(nls_np_norm[:,None]+1e-5)
    print("processing pointcloud......")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd.normals = o3d.utility.Vector3dVector(nls_np)
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)     #先点云下采样，保证点云分布均匀
    # _, index = pcd.remove_statistical_outlier(nb_neighbors = 30,std_ratio= 7.0)   #然后点云统计滤波
    # pcd = pcd.select_by_index(index)
    pcd = pcd.voxel_down_sample(down_voxel_size)     #再点云下采样，得到最终的点云
    # print("point number: %d"%len(pcd.points))
    
    pcd_torch=torch.tensor(np.asarray(pcd.points), device = "cpu").float()
    print("processing pointcloud successful")
    print("insert to spatial hash grid.....")
    svh.insert(pcd_torch)     #哈希表插入操作 

    local_sdfs.update_in_svh(pcd)     # 向自己的哈希网格中更新局部SDF
    local_sdfs.prepare_for_optimization()   # 为优化做准备

    print("delete hash voxel......")
    n_connect = 20
    svh.delete_voxel(n_connect)    #删除没有通过检验的体素
    print("delete hash voxel successful")


## 射线与哈希表中的voxel进行相交测试（ray_d是世界坐标系中没有归一化的射线方向）
@torch.no_grad()
def ray_intersection_svh(svh, ht_info, ray_o, ray_d, dminmax,    #当sur_behind_dis>0且gt_depth!=None时，执行深度截断
    n_max = 5, 
    sur_behind_dis = 0.0, 
    gt_depth = None
    ):
    G = 128 #256   # 128   # 哈希表太大出bug时，把这个设小点
    N = ray_o.shape[0]  #N为光线数
    K = int(np.ceil(N / G))
    H = K * G
    if H > N:
        ray_o = torch.cat([ray_o, ray_o[: H - N]], 0)   
        ray_d = torch.cat([ray_d, ray_d[: H - N]], 0)
        dminmax= torch.cat([dminmax, dminmax[: H - N]], 0)
    ray_o = ray_o.reshape(G, K, 3).contiguous()     #需要将数组变为连续的
    ray_d = ray_d.reshape(G, K, 3).contiguous()
    dminmax = dminmax.reshape(G, K, 2).contiguous()
    ht_info = ht_info.expand(G, *ht_info.size()).contiguous()

    inds, min_depth, max_depth = svh.ray_intersect(ray_o.float(), ray_d.float(), dminmax.float(), ht_info.int(), n_max)
    inds = inds.reshape(H, -1)
    min_depth = min_depth.reshape(H, -1)
    max_depth = max_depth.reshape(H, -1)
    if H > N:
        inds = inds[:N]
        min_depth = min_depth[:N]
        max_depth = max_depth[:N]
    # print(222)
    max_distance = 99999.0
    min_depth.masked_fill_(inds.eq(-1), max_distance)
    max_depth.masked_fill_(inds.eq(-1), max_distance)

    if(gt_depth != None and sur_behind_dis > 0):     #采样截断到depth_clip处
        depth_clip = gt_depth + sur_behind_dis
        mask_clip = min_depth > depth_clip[:, None]
        inds[mask_clip] = -1
        #1. 由于不连通体素的删除、分配体素网格时深度图缩放和间隔加载，导致存在有深度值但没有分配体素网格的情况，
        # 会导致第一个相交网格点的深度大于真实深度值，需要排除这种情况
        #2. 深度值为0的射线也会与网格相交，也需要排除这种情况
        no_hits = min_depth[:, 0]>gt_depth   
        inds[no_hits] = -1

        ids_0 = torch.arange(max_depth.shape[0]).long()
        ids_1 = inds.ne(-1).sum(-1) -1
        mask = max_depth[ids_0, ids_1] > depth_clip
        max_depth[ids_0, ids_1][mask] = depth_clip[mask]
        min_depth.masked_fill_(inds.eq(-1), max_distance)
        max_depth.masked_fill_(inds.eq(-1), max_distance)

    # remove all points that completely miss the object
    max_hits = torch.max(inds.ne(-1).sum(-1))
    min_depth = min_depth[..., :max_hits]
    max_depth = max_depth[..., :max_hits]
    inds = inds[..., :max_hits]
    hits = inds.ne(-1).any(-1)

    intersection_outputs = {
        "min_depth": min_depth,
        "max_depth": max_depth,
        "intersected_voxel_idx": inds,
    }

    return intersection_outputs, hits

#逆变换采样
def inverse_cdf_sampling_svh(svh, pts_idx,  min_depth, max_depth, probs, steps, fixed_step_size=-1, deterministic=False):
    G, N, P = 200, pts_idx.size(0), pts_idx.size(1)
    H = int(np.ceil(N / G)) * G

    if H > N:
        pts_idx = torch.cat([pts_idx, pts_idx[:1].expand(H - N, P)], 0)
        min_depth = torch.cat(
            [min_depth, min_depth[:1].expand(H - N, P)], 0)
        max_depth = torch.cat(
            [max_depth, max_depth[:1].expand(H - N, P)], 0)
        probs = torch.cat([probs, probs[:1].expand(H - N, P)], 0)
        steps = torch.cat([steps, steps[:1].expand(H - N)], 0)

    pts_idx = pts_idx.reshape(G, -1, P)
    min_depth = min_depth.reshape(G, -1, P)
    max_depth = max_depth.reshape(G, -1, P)
    probs = probs.reshape(G, -1, P)
    steps = steps.reshape(G, -1)

    # pre-generate noise
    max_steps = steps.ceil().long().max() + P
    noise = min_depth.new_zeros(*min_depth.size()[:-1], max_steps)
    if deterministic:
        noise += 0.5
    else:
        noise = noise.uniform_().clamp(min=0.001, max=0.999)  # in case
    
    # call cuda function
    chunk_size = 4 * G  # to avoid oom?
    results = [
        svh.inverse_cdf_sampling(
            pts_idx[:, i: i + chunk_size].contiguous(),
            min_depth.float()[:, i: i + chunk_size].contiguous(),
            max_depth.float()[:, i: i + chunk_size].contiguous(),
            noise.float()[:, i: i + chunk_size].contiguous(),
            probs.float()[:, i: i + chunk_size].contiguous(),
            steps.float()[:, i: i + chunk_size].contiguous(),
            fixed_step_size,
        )
        for i in range(0, min_depth.size(1), chunk_size)
    ]

    sampled_idx, sampled_depth, sampled_dists = [
        torch.cat([r[i] for r in results], 1) for i in range(3)
    ]
    sampled_depth = sampled_depth.type_as(min_depth)
    sampled_dists = sampled_dists.type_as(min_depth)

    sampled_idx = sampled_idx.reshape(H, -1)
    sampled_depth = sampled_depth.reshape(H, -1)
    sampled_dists = sampled_dists.reshape(H, -1)
    if H > N:
        sampled_idx = sampled_idx[:N]
        sampled_depth = sampled_depth[:N]
        sampled_dists = sampled_dists[:N]

    max_len = sampled_idx.ne(-1).sum(-1).max()
    sampled_idx = sampled_idx[:, :max_len]
    sampled_depth = sampled_depth[:, :max_len]
    sampled_dists = sampled_dists[:, :max_len]

    return sampled_idx, sampled_depth, sampled_dists

## 在光线与voxel相交的部分上采样
@torch.no_grad()
def ray_sample_svh(svh, intersection_outputs, ray_o, ray_d, 
    step_size=0.01, 
    fixed=False, 
    n_surf_samples=0, 
    s_dev = 0.01,                 #表面采样的标准差
    sur_behind_dis=0,
    gt_depth = None
):
    dists = (
        intersection_outputs["max_depth"] -
        intersection_outputs["min_depth"]
    ).masked_fill(intersection_outputs["intersected_voxel_idx"].eq(-1), 0)
    intersection_outputs["probs"] = dists / dists.sum(dim=-1, keepdim=True)     #这里是使用与每个voxel相交线的长度来充当概率，长度越大概率越大
    intersection_outputs["steps"] = dists.sum(-1) / step_size

    sampled_idx, sampled_depth, sampled_dists = inverse_cdf_sampling_svh(  
        svh,  
        intersection_outputs["intersected_voxel_idx"],
        intersection_outputs["min_depth"],
        intersection_outputs["max_depth"],
        intersection_outputs["probs"],
        intersection_outputs["steps"], -1, fixed)
    # print(sampled_depth.shape)
    sampled_dists = sampled_dists.clamp(min=0.0)
    MAX_DEPTH = 99999.0
    sampled_depth.masked_fill_(sampled_idx.eq(-1), MAX_DEPTH)
    sampled_dists.masked_fill_(sampled_idx.eq(-1), 0.0)

    sample_mask = sampled_idx.ne(-1)

    if(gt_depth != None and n_surf_samples > 0 and sur_behind_dis > 0):      #表面采样
        surface_z_vals = gt_depth
        offsets = torch.normal(
            torch.zeros(gt_depth.shape[0], n_surf_samples - 1), s_dev
        ).to(sampled_depth.device)
        near_surf_z_vals = gt_depth[:, None] + offsets

        min_depth = torch.zeros_like(surface_z_vals).to(surface_z_vals.device)   #表面处的深度值截断
        max_depth = surface_z_vals + sur_behind_dis
        near_surf_z_vals = torch.clamp(
            near_surf_z_vals,
            min_depth[:, None],
            max_depth[:, None],
        )

        sampled_depth = torch.cat(
            (surface_z_vals[:, None], near_surf_z_vals, sampled_depth), dim=1)
        mask_expand = torch.ones(near_surf_z_vals.shape[0], near_surf_z_vals.shape[1]+1).bool().to(sample_mask.device)
        sample_mask = torch.cat((mask_expand, sample_mask), dim=1)

    pts_W = ray_o[:, None, :] + ray_d[:, None, :]  * sampled_depth[:, :, None]

    return pts_W, sampled_depth, sampled_dists, sample_mask


##在哈希voxel中采样
def sample_points_svh(
    svh,
    ht_info,
    pc_batch,                                              #世界坐标系下的点云
    T_WC_batch,
    bounds_extents,                                 #orient包围盒的长宽高
    inv_bounds_transform,                   #世界坐标系到包围盒坐标系的变换
    n_rays = 256,                                         #每帧预计采样的光线数
    n_max = 10,                                           #光线相交的最大网格数
    sur_behind_dis = 0.05,                     #采样时物体表面之后的采样距离
    n_surf_samples=30,                          #采样时在物体表面采样的个数
    s_dev = 0.01,                                         #表面采样的标准差
    step_size_sdf=0.05,                           #采样时沿光线采样的间距
    device = 'cuda'
):
    with torch.set_grad_enabled(False):
        n_frames = pc_batch.shape[0]
        n_rays_per_fm = pc_batch.shape[1]
        indices_b = torch.arange(n_frames, device=device)
        indices_b = indices_b.repeat_interleave(n_rays)
        total_rays = n_frames * n_rays
        indices_n =  torch.randint(0, n_rays_per_fm, (total_rays,), device=device)
        
        pc_sample = pc_batch[indices_b, indices_n]
        T_sample = T_WC_batch[indices_b]
        ray_o_W = T_sample[:, :3, 3]
        t_sample = (pc_sample - ray_o_W).norm(dim = -1)
        ray_d_W = F.normalize((pc_sample - ray_o_W),dim = -1)    #单位化的射线方向

        #得到射线在orient包围盒最小最大值
        tminmax, ray_mask = ray_intersect(bounds_extents, inv_bounds_transform, ray_o_W, ray_d_W)
        pc_sample = pc_sample[ray_mask]      #筛选与包围盒相交的射线
        T_sample = T_sample[ray_mask]
        t_sample = t_sample[ray_mask]
        ray_o_W = ray_o_W[ray_mask]
        ray_d_W = ray_d_W[ray_mask]
        tminmax = tminmax[ray_mask]

        #光线与哈希表中voxel相交测试（sdf）
        intersections_sdf, hits_sdf = ray_intersection_svh(svh, ht_info, ray_o_W, ray_d_W, tminmax, n_max = n_max, sur_behind_dis=sur_behind_dis, gt_depth = t_sample)   
        
        intersections_sdf = {    #这一步是利用掩膜提取字典中的每个元素
            name: outs[hits_sdf]
            for name, outs in intersections_sdf.items()
        }

        T_sample_sdf = T_sample[hits_sdf]
        ray_o_W_sdf = ray_o_W[hits_sdf]    #选择与哈希表中voxel相交的射线
        ray_d_W_sdf = ray_d_W[hits_sdf]
        t_sample_sdf = t_sample[hits_sdf]
        pc_sdf, z_vals_sdf, _, sample_mask_sdf = ray_sample_svh(svh, intersections_sdf, ray_o_W_sdf, ray_d_W_sdf,  
            step_size=step_size_sdf, fixed=False, n_surf_samples = n_surf_samples, s_dev = s_dev, sur_behind_dis=sur_behind_dis, 
            gt_depth = t_sample_sdf)

        sample_pts = {
            "pc_sdf": pc_sdf,
            "z_vals_sdf": z_vals_sdf,      #这里的z_vals_sdf为是延光线的距离，不是深度值，因为前面函数中的方向参数都为单位向量
            "t_sample_sdf": t_sample_sdf,      #无深度为0的值
            "ray_d_W_sdf": ray_d_W_sdf,
            "T_sample_sdf": T_sample_sdf,
            "sample_mask_sdf": sample_mask_sdf,
        } 

        return sample_pts

# 计算损失
def compute_loss(
    neural_map,
    sample,
    iter,
    trunc_distance,
    trunc_weight,
    add_scale_loss,
):
    pc = sample["pc_sdf"]
    z_vals = sample["z_vals_sdf"]
    sample_mask = sample["sample_mask_sdf"]
    t_sample_sdf = sample["t_sample_sdf"]

    #这一步是为了求得sdf真值，因此设为不可导
    with torch.set_grad_enabled(False):
        bounds = t_sample_sdf[:, None] - z_vals

    pc = pc[sample_mask]
    z_vals = z_vals[sample_mask]
    bounds = bounds[sample_mask]        
    
    pc.requires_grad_()   #设置三维坐标可导
    sdf, _ = neural_map.query_sdf(pc)  #输入采样点三维坐标，利用局部SDF，预测sdf值
    sdf_grad = network.gradient(pc, sdf)   #计算预测的sdf值对三维坐标的导数
    
    loss = 0
     # compute loss
    free_space_loss_mat = sdf - torch.ones(sdf.shape, device=sdf.device) * trunc_distance
    trunc_loss_mat = sdf- bounds
    free_space_ixs = bounds > trunc_distance
    free_space_loss_mat[~free_space_ixs] = 0.
    trunc_loss_mat[free_space_ixs] = 0.
    sdf_loss_mat = free_space_loss_mat + trunc_loss_mat
    sdf_loss_mat = torch.abs(sdf_loss_mat)
     
    sdf_loss_mat[~free_space_ixs] *= trunc_weight
    invalid_mask = torch.isnan(sdf_loss_mat)
    sdf_loss = sdf_loss_mat[~invalid_mask].mean() 
    # print("sdf_loss: %f"%sdf_loss.item())
    sdf_loss = sdf_loss / trunc_distance
    loss = loss + sdf_loss
    
    # 将基SDF初始化为平面 ，warm up
    if iter < 500:
        length = 3-3/500*iter
        x = y = z = torch.linspace(-length, length, 32)
        zz, yy, xx = torch.meshgrid(z, y, x)
        sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float()
        sampled_xyz = sampled_xyz.reshape(-1,3).to("cuda")
        label_sdf = sampled_xyz[:,2]
        sdf_regularize = neural_map.query_in_basis_sdf(sampled_xyz)
        sdf_regularize_loss = torch.abs(sdf_regularize - label_sdf).mean()
        # print("sdf_regularize_loss: %f"%sdf_regularize_loss.item())
        loss = loss + sdf_regularize_loss
    
    if add_scale_loss:
        scale_weight = 0.1
        updata_mask = neural_map.neural_points.updata_mask
        scale_normal = torch.exp(neural_map.neural_points.scales[updata_mask])[:,2]
        scale_loss = torch.abs(scale_normal - neural_map.neural_points.resolution / 3).mean() 
        scales_mean = torch.exp(neural_map.neural_points.scales[updata_mask]).mean(0)
        # print("scales_mean: %.2f,%.2f,%.2f"%(scales_mean[0].item(), scales_mean[1].item(), scales_mean[2].item()))
        scale_loss = scale_weight*scale_loss
        loss =  loss + scale_loss

    eik_weight = 0.02
    eik_loss_mat = torch.abs(sdf_grad.norm(2, dim=-1) - 1)
    eik_loss_mat = eik_loss_mat[~free_space_ixs]
    invalid_mask = torch.isnan(eik_loss_mat)
    eik_loss = eik_loss_mat[~invalid_mask].mean()
    # print("eik_loss: %f"%eik_loss.item())
    eik_loss = eik_loss * eik_weight
    loss = loss + eik_loss

    return  loss


#在哈希网格中创建mesh，grid_res为分辨率(个数)
def create_mesh_svh(ht_info, vox_size, neural_map, grid_res, chunk_size, mesh_min_nn = 6, save_path = None, device = "cuda"):
    #得到voxel中心
    vox_coords = ht_info[:, :3]
    inval_val = 999999    #无效坐标值，与c++类中对应
    vox_coords = vox_coords[vox_coords[:, 0] != inval_val]
    vox_center = (vox_coords+0.5) * vox_size    #得到体素中心坐标
    
    vox_coords_min, _ = vox_coords.min(dim = 0)
    vox_coords_max, _ = vox_coords.max(dim = 0)
    vox_coords_res = vox_coords_max - vox_coords_min
    vox_coords_local = vox_coords - vox_coords_min[None, :]
    vox_idx = vox_coords_local[:,2] * vox_coords_res[0] * vox_coords_res[1] + vox_coords_local[:,1] * vox_coords_res[0] + vox_coords_local[:,0]   # 计算每个voxel的id
    vox_idx = vox_idx.long()
    
    # 构建spatial_vox_size*spatial_vox_size的二维网格
    spatial_vox_size = 200
    bbox_min = (torch.min(vox_center, dim=0).values-1).int()
    bbox_max = (torch.max(vox_center, dim=0).values+1).int()
    bbox_len = bbox_max - bbox_min
    vox_center_axis1 = None
    vox_center_axis2 = None
    if torch.argmin(bbox_len).item() == 0:
        list_a = torch.arange(bbox_min[1], bbox_max[1], spatial_vox_size)
        list_b = torch.arange(bbox_min[2], bbox_max[2], spatial_vox_size)
        vox_center_axis1 = vox_center[:, 1]
        vox_center_axis2 = vox_center[:, 2]
    elif torch.argmin(bbox_len).item() == 1:
        list_a = torch.arange(bbox_min[0], bbox_max[0], spatial_vox_size)
        list_b = torch.arange(bbox_min[2], bbox_max[2], spatial_vox_size)
        vox_center_axis1 = vox_center[:, 0]
        vox_center_axis2 = vox_center[:, 2]
    elif torch.argmin(bbox_len).item() == 2:
        list_a = torch.arange(bbox_min[0], bbox_max[0], spatial_vox_size)
        list_b = torch.arange(bbox_min[1], bbox_max[1], spatial_vox_size)
        vox_center_axis1 = vox_center[:, 0]
        vox_center_axis2 = vox_center[:, 1]
    grid_a, grid_b = torch.meshgrid(list_a, list_b)
    grid = torch.stack([grid_a, grid_b], dim = -1)
    # print(grid.shape)
    grid = grid.reshape(-1, 2)
    
    vox_idx_max = vox_idx.max()
    # print(vox_idx_min, vox_idx_max)
    # 在每个网格块中再分批处理
    vox_num_all = []
    kk = 0
    for k in range(grid.shape[0]):
        vox_min = grid[k]
        vox_max = grid[k] + spatial_vox_size
        mask = (vox_center_axis1 >= vox_min[0]) & (vox_center_axis1 < vox_max[0]) & (vox_center_axis2 >= vox_min[1]) & (vox_center_axis2 < vox_max[1])
        if (mask.sum() == 0):
            continue
        vox_idx[mask] += (kk * vox_idx_max)
        vox_num_all.append(len(vox_idx[mask]))
        kk += 1
    
    _, sorted_id = torch.sort(vox_idx)
    vox_coords = vox_coords[sorted_id]   # 排序过后的体素坐标
    vox_center = (vox_coords+0.5) * vox_size    #得到体素中心坐标
    
    #得到每个voxel的sdf_volumn
    sdf_grid = None
    mc_mask_grid = None
    n_voxel = vox_center.size(0)
    m = 0
    n_voxel_t = vox_num_all[0]
    id_s = 0
    id_e = id_s+chunk_size
    while(id_s < n_voxel):
        if(id_e >= n_voxel_t):
            id_e = n_voxel_t
            m += 1
            if(m < len(vox_num_all)):
                n_voxel_t += vox_num_all[m]
        
        vox_center_select = vox_center[id_s:id_e]
        
        start = -0.5
        end = 0.5  
        x = y = z = torch.linspace(start, end, grid_res)
        xx, yy, zz = torch.meshgrid(x, y, z)
        sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float().to(device)
        sampled_xyz *= vox_size
        sampled_xyz = sampled_xyz.reshape(1, -1, 3) + vox_center_select.unsqueeze(1)
        _shape = sampled_xyz.shape
        
        sdf, nn_count = neural_map.query_sdf(sampled_xyz)    # 局部SDF地图
        
        sdf = sdf.reshape(sampled_xyz.shape[0], grid_res, grid_res, grid_res)
        if sdf_grid == None:
            sdf_grid = sdf.detach().cpu()
        else:
            sdf_grid = torch.cat([sdf_grid, sdf.detach().cpu()], dim = 0)
            
        mask_mc = (nn_count >= mesh_min_nn)  
        mask_mc = mask_mc.reshape(_shape[0], grid_res, grid_res, grid_res)
        if mc_mask_grid == None:
            mc_mask_grid = mask_mc.cpu()
        else:
            mc_mask_grid = torch.cat([mc_mask_grid, mask_mc.cpu()], dim = 0)
        
        id_s = id_e
        id_e = id_s+chunk_size
    
    #对每个sdf_volume执行marching cube
    res = 1.0 / (sdf_grid.shape[1] - 1)
    spacing = [res, res, res]
    num_verts = 0
    total_verts = []
    total_faces = []
    n_voxel_total = vox_center.size(0)
    for i in range(n_voxel_total):
        sdf_volume = sdf_grid[i].numpy()
        mc_mask = mc_mask_grid[i].numpy()
        #已在模块内部进行了修改，若提取不到mesh，则返回None
        verts, faces, _, _ = skimage_measure.marching_cubes(sdf_volume, 0, spacing=spacing, mask=mc_mask)   
        if verts is None:
            continue
        verts -= 0.5
        verts *= vox_size
        verts += vox_center[i].detach().cpu().numpy()
        faces += num_verts
        num_verts += verts.shape[0]
        total_verts += [verts]
        total_faces += [faces]
        
    total_verts = np.concatenate(total_verts)
    total_faces = np.concatenate(total_faces)
    
    #构建mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(total_verts)
    mesh.triangles = o3d.utility.Vector3iVector(total_faces)
    mesh.compute_vertex_normals()
    
    if save_path != None:
        o3d.io.write_triangle_mesh(save_path, mesh)

    return mesh
            

def freeze_model(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def freeze_neural_points(neural_points):
    neural_points.neural_points.requires_grad = False
    neural_points.orientations.requires_grad = False
    neural_points.scales.requires_grad = False
    
def unfreeze_neural_points(neural_points):
    neural_points.neural_points.requires_grad = True
    neural_points.orientations.requires_grad = True
    neural_points.scales.requires_grad = True
