import os, argparse
import numpy as np
import torch
import torch.optim as optim

import open3d as o3d
import yaml
import pypose as pp

import dataset, network, main_util
from local_sdf import LocalSDF
from tqdm import *

torch.classes.load_library("Thirdparty/sparse_hash/build/libsvh.so")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default = None)
    args = parser.parse_args()

    config_file = args.conf

    f = open(config_file,encoding='utf-8')
    cfg = yaml.load(f.read(),Loader=yaml.FullLoader)
    print(cfg)

    #参数设置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #文件路径
    workspace = cfg["Dataset"] ["workspace"]
    root_dir = workspace + cfg["Dataset"] ["data_dir"] 
    save_path = cfg["Dataset"] ["out_dir"]    #网络模型和mesh模型的保存路径
    if(not os.path.exists(save_path)):
        os.mkdir(save_path)
    ht_path = save_path+"ht.pth"

    #加载数据
    min_range = cfg["DataLoader"] ["min_range"]
    max_range = cfg["DataLoader"] ["max_range"]
    sor_nn = cfg["DataLoader"] ["sor_nn"]
    sor_std = cfg["DataLoader"] ["sor_std"]
    use_filter = cfg["DataLoader"] ["use_filter"]

    pc_dir = root_dir +   cfg["Dataset"] ["pc_dir"]          
    pose_file = root_dir +   cfg["Dataset"] ["pose_file"]
    calib_file = None
    if "calib_file" in cfg["Dataset"]:
        calib_file = root_dir +   cfg["Dataset"] ["calib_file"]
    data_set = dataset.LiDARDataset(pc_dir, calib_file, pose_file, min_range, max_range, 
                sor_nn, sor_std, use_filter)
    
    #--------------------------------------------------------------------------------------------------------------------------------

    #分配体素
    voxel_size = cfg["HashTable"] ["voxel_size"]          #体素网格的大小
    ht_size = cfg["HashTable"] ["ht_size"]                       #哈希表的大小
    skip_load = cfg["HashTable"] ["skip_load"]        

    n_dataset = len(data_set)
    n_load_total = n_dataset//skip_load   
    ind = np.linspace(0, n_dataset, n_load_total, dtype = int, endpoint= False)

    svh = torch.classes.svh.HashTable(voxel_size, ht_size)    #初始化哈希表
    local_sdfs = LocalSDF(            # 初始化local sdf
        resolution = cfg["LocalSDF"]["resolution"],
        query_nn_k = cfg["LocalSDF"]["query_nn_k"],
        gss_vox_size = cfg["LocalSDF"]["gss_vox_size"]
    )    
    
    inval_val = cfg["HashTable"] ["inval_val"]     #无效坐标值，与c++类中对应
    ht_info = None
    lcp_index=None
    lcp_array = None
    if(os.path.exists(ht_path)):
        print("load hash info and local cloud point......")
        ht_info, lcp_index, lcp_array = torch.load(ht_path)
        vox_coords = ht_info[:, :3]
        vox_coords = vox_coords[vox_coords[:, 0] != inval_val]
        vox_center = (vox_coords+0.5) * voxel_size    #得到体素中心坐标
        svh.insert(vox_center)    # 向svh中插入，构建svh
    else:
        main_util.allocate_localsdfs_in_svh(svh, data_set, ind, local_sdfs,
            res_scale = cfg["HashTable"] ["res_scale"] ,
            down_voxel_size= cfg["HashTable"] ["down_vox_size"])       
        ht_info, lcp_index, lcp_array = svh.get_ht_info()   #获取哈希表信息,，lcp_index为局部点云的索引信息，lcp_array为点云数组
        torch.save((ht_info, lcp_index, lcp_array), ht_path)   #保存哈希表信息与局部点云
        print("hash grid info has been saved!")
    vox_coord = (ht_info[:, :3]+0.5) * voxel_size
    vox_coord = vox_coord[torch.where(ht_info[:, 0]!=inval_val)].numpy()
    # 将ht_info与lcp_index传入gpu中
    ht_info = ht_info.to(device)
    lcp_index = lcp_index.to(device)
    
    #-------------------------------------------------------------------------------------------------------------------------------------
    # 计算场景的oriented_bounding_box
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vox_coord)
    obb = pcd.get_oriented_bounding_box() 
    bounds_extents_np = obb.extent 
    T_extent_to_scene = np.eye(4) 
    T_extent_to_scene[:3, :3] = obb.R  
    T_extent_to_scene[:3, 3] = obb.center  
    T_extent_to_scene = np.linalg.inv(T_extent_to_scene)

    bound_scale = 1.1    #适当扩大bound
    bounds_extents_np = bounds_extents_np * bound_scale
    bounds_extents = torch.from_numpy(bounds_extents_np).float().to(device)
    inv_bounds_transform = torch.from_numpy(T_extent_to_scene).float().to(device)

    #----------------------------------------------------------------------------------------------------------------------------

   #加载网络，设置优化器
    neural_map = network.NeuralMap(
        local_sdfs,
        num_layers=cfg["Network"] ["num_layers"],
        hidden_dim=cfg["Network"] ["hidden_dim"],
    )

    n_iter_start = 0
    model_path = save_path+"model.pth"
    if(os.path.exists(model_path)):
        dic_params= torch.load(model_path)
        net_params = dic_params["net_params"]
        n_iter_start = dic_params["n_iter"]
        local_sdfs.init_from_preload_params(net_params)             
        neural_map.load_state_dict(net_params)
    else:
        dic_params = {"net_params": neural_map.state_dict(), "n_iter": 0}
        torch.save(dic_params,  model_path)
    neural_map= neural_map.to(device)

    n_epoch = cfg["Train"] ["n_epoch"]                         #epoch次数
    n_step = cfg["Train"] ["n_step"]                         #每个epoch训练的次数
    params = [
            {'params': neural_map.sdf_net.parameters(),  "name": "sdf_net"},
            {'params': [local_sdfs.positions], "name": "positions"},
            {'params': [local_sdfs.rotations], "name": "rotations"},
            {'params': [local_sdfs.scalings], "name": "scalings"}
        ]
    optimizer = optim.Adam(params, lr=5e-3)
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: cfg["Train"] ["lr_ratio"] ** min(iter / (n_epoch * n_step), 1))
    lr_scheduler = scheduler(optimizer)
    
    # 加载数据，构建缓存
    print("loading data ......")
    buffer_path = save_path+"data_buffer.pth"
    pc_batch, normal_batch, T_batch = None, None, None
    if(os.path.exists(buffer_path)):
        data_buffer= torch.load(buffer_path)
        pc_batch, normal_batch, T_batch = data_buffer["pc_batch"], data_buffer["normal_batch"], data_buffer["T_batch"] 
    else:
        pc_batch, normal_batch, T_batch = main_util.load_data_buffer(data_set, is_est_normal=False)
        data_buffer = {"pc_batch": pc_batch, "normal_batch": normal_batch, "T_batch": T_batch}
        torch.save(data_buffer,  buffer_path)
        print("frame buffer data has been saved!")
    
    # 每interval_skip帧选择一帧
    data_idx = torch.arange(0, len(pc_batch), cfg["Train"] ["interval_skip"])
    pc_batch = pc_batch[data_idx]
    if normal_batch is not None:
        normal_batch = normal_batch[data_idx]
    T_batch = T_batch[data_idx]
    
    # 初始化可视化器
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    render_option = visualizer.get_render_option()
    render_option.point_size = 2 
    vis_neural_map = o3d.geometry.PointCloud()
#--------------------------------------------------------------------------------------------------------------------------------
    #训练  
    n_views_total = cfg["Train"] ["n_views_total"]                             #加载的帧数
    n_views_select = cfg["RaySample"] ["n_views_select"]           #每个step进行采样的帧数
    accum_pts = None
    depth_pose = None
    propatation_weights = None
    
    pbar = tqdm(total=n_epoch*n_step, leave=False)
    pbar.update(n_iter_start+1)
    for epoch in range(n_epoch):
        n_dataset = len(pc_batch)
        indice_s = np.random.randint(0, n_dataset, 1)
        indices = np.arange(indice_s, indice_s+n_views_total) % n_dataset   # 选择连续的n_views_total帧
        pc_batch_, T_batch_ = pc_batch[indices], T_batch[indices]    # 使用缓冲区的数据
        for iter in range(n_step):
            n_iter_current = epoch * n_step + iter
            if n_iter_current < n_iter_start+1:    # 这里是为了继续训练,而不是从头开始
                lr_scheduler.step()
                continue
            n_views_load = pc_batch_.shape[0]   #在get_data中可能会摒弃点个数较少的点云帧，会导致n_views_total有变化，所以这里更新一下
            frame_ids = (torch.rand(n_views_select)*n_views_load).long()
            pc_batch_select = pc_batch_[frame_ids].to(device)
            T_batch_select = T_batch_[frame_ids].to(device)

            sample_pts = main_util.sample_points_svh(
                svh,
                ht_info,
                pc_batch_select,                                                                                    #世界坐标系下的点云
                T_batch_select,
                bounds_extents,                                                                                      #orient包围盒的长宽高
                inv_bounds_transform,                                                                       #世界坐标系到包围盒坐标系的变换，为了计算光线与oriented包围盒交点的深度，光线截止到交点
                n_rays = cfg["RaySample"] ["n_rays"],                                          #预计每张图像上采样的光线数
                n_max = cfg["RaySample"] ["n_max_interset"],                       #光线相交的最大网格数
                sur_behind_dis = cfg["RaySample"] ["sur_behind_dis"],     #采样时物体表面之后的采样距离
                n_surf_samples=cfg["RaySample"] ["n_surf_samples"],     #采样时在物体表面采样的个数
                s_dev=cfg["RaySample"] ["s_dev"],                                               #表面采样的标准差
                step_size_sdf=cfg["RaySample"] ["step_size_sdf"],               #采样时沿光线采样的间距
                device = device
            )
            loss = main_util.compute_loss(
                neural_map,
                sample_pts,
                iter = n_iter_current,
                trunc_distance = cfg["Loss"] ["trunc_distance"],      #截断距离
                trunc_weight = cfg["Loss"] ["trunc_weight"],            #截断距离内的sdf_loss权重
                add_scale_loss= cfg["LocalSDF"]["add_scale_loss"], 
            )
            
            loss.backward()   
            
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
            
            if(n_iter_current > cfg["Train"]["freeze_after_iters"]):      # N次迭代后固定住decoder的参数
                main_util.freeze_model(neural_map.sdf_net)

            neural_map.vis_basis_sdf(resolution=256, vis="xz")    # 可视化basis sdf
            if cfg["LocalSDF"]["exe_densitify"]:
                local_sdfs.add_densification_stats()
                if (n_iter_current+1)>=1000 and (n_iter_current+1)%cfg["LocalSDF"]["n_iters_densitify"]==0:
                    local_sdfs.densify_and_prune(cfg, iter = n_iter_current, optimizer=optimizer, neural_map=neural_map)
            
            optimizer.step()            
            optimizer.zero_grad()
            lr_scheduler.step()
            
            # 更新可视化器
            every_update = 20 #200   # 可调大些，降低可视化更新频率，运行效率
            if (n_iter_current-1)%every_update==0 or n_iter_current==n_iter_start+1:
                positions_np = local_sdfs.positions.detach().cpu().numpy()
                normals_np = pp.so3(local_sdfs.rotations).Exp().matrix()[..., 2].cpu().detach().numpy().astype(np.float64)      
                normals_color = ((normals_np + 1)/2)
                vis_neural_map.points = o3d.utility.Vector3dVector(positions_np)
                vis_neural_map.colors = o3d.utility.Vector3dVector(normals_color)   # 法向着色
                
                if n_iter_current==n_iter_start+1:
                    visualizer.add_geometry(vis_neural_map)
                else:
                    visualizer.update_geometry(vis_neural_map) 

            visualizer.poll_events()
            visualizer.update_renderer()

            n_every_save_iters = cfg["Save"] ["n_every_save_iters"]
            if (n_iter_current+1) % n_every_save_iters == 0:
                #保存模型和mesh
                print('Saving model and mesh in %s'%save_path)
                mesh_save_path = save_path+"mesh_%d.ply"%(n_iter_current+1)
                model_save_path = save_path+"model_%d.pth"%(n_iter_current+1)
                dic_params = {"net_params": neural_map.state_dict(), "n_iter": n_iter_current}
                torch.save(dic_params,  model_save_path)
                mesh_ = main_util.create_mesh_svh(ht_info, voxel_size, neural_map, grid_res = cfg["Save"] ["grid_res"], 
                    chunk_size = 256, mesh_min_nn= cfg["Save"] ["mesh_min_nn"], save_path = mesh_save_path, device = device)
                
    pbar.close()
