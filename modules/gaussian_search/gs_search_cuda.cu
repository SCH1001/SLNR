#include "gs_search_cuda.h"
#include <Eigen/Eigen>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
#define N_SIGMA 3

__forceinline__ __device__ void _ceil(float x, float y, int& ceil)
{
    float ceil_ = x/y;
    int integer = (int)ceil_;
    ceil = (ceil_ - integer)>0 ? integer+1 : integer;
}

// voxel_grid为grid的分辨率，p_lb和p_ru均为以bounding左下角为原点的局部坐标
__forceinline__ __device__ void getBox(const Eigen::Vector3f p_lb, const Eigen::Vector3f p_ru, uint3& box_min, uint3& box_max, dim3 grid, float vox_size)
{
    box_min = {
        min(grid.x, max((int)0, (int)(p_lb[0] / vox_size))),
        min(grid.y, max((int)0, (int)(p_lb[1] / vox_size))),
        min(grid.z, max((int)0, (int)(p_lb[2] / vox_size)))
    };
    int ceil_x, ceil_y, ceil_z;
    _ceil(p_ru[0], vox_size, ceil_x);
    _ceil(p_ru[1], vox_size, ceil_y);
    _ceil(p_ru[2], vox_size, ceil_z);
    box_max = {
        min(grid.x, max((int)0, ceil_x)),
        min(grid.y, max((int)0, ceil_y)),
        min(grid.z, max((int)0, ceil_z))
    };
}

//--------------------------------------------------
//说明：根据给定的矩形边界，查找与矩形相交的gaussian，返回掩码,核函数
//作者：Mr_Shi
//--------------------------------------------------
__global__ void find_gs_by_boundingbox_kernel(int n_gs,
        float x_min, float y_min, float z_min,
        float x_max, float y_max, float z_max,
        const Eigen::Vector3f* gs_centers,
        const Eigen::Vector3f* gs_rotvecs,
        const Eigen::Vector3f* gs_scales,
        bool* inside_mask)   //返回结果
{
    auto idx = cg::this_grid().thread_rank();   // 每个gaussian为一个线程
    if (idx >= n_gs)
        return;
    Eigen::Vector3f center = gs_centers[idx];
    Eigen::Vector3f rotvec = gs_rotvecs[idx];
    Eigen::Vector3f scale = gs_scales[idx]*N_SIGMA;   // 取3*scale大小的范围

    Eigen::Vector3f box_vertex[8]={   // gs范围的八个顶点
        Eigen::Vector3f(-scale[0],-scale[1],-scale[2]), Eigen::Vector3f(-scale[0],-scale[1],scale[2]),
        Eigen::Vector3f(-scale[0],scale[1],-scale[2]), Eigen::Vector3f(-scale[0],scale[1],scale[2]),
        Eigen::Vector3f(scale[0],-scale[1],-scale[2]), Eigen::Vector3f(scale[0],-scale[1],scale[2]),
        Eigen::Vector3f(scale[0],scale[1],-scale[2]), Eigen::Vector3f(scale[0],scale[1],scale[2])
    };
    for(int i = 0; i < 8; i++)
    {
        Eigen::Vector3f pt = box_vertex[i];

        float norm = rotvec.norm();
        Eigen::Vector3f dir = rotvec.normalized();

        Eigen::AngleAxisf rotation_vector;
        if(norm < 1e-7) rotation_vector = Eigen::AngleAxisf::Identity();
        else rotation_vector = Eigen::AngleAxisf(norm, dir);

        pt = rotation_vector.matrix() * pt + center;

        if(pt[0]>x_min && pt[1]>y_min && pt[2]>z_min && pt[0]<x_max && pt[1]<y_max && pt[2]<z_max)
        {
            inside_mask[idx] = 1;
            break;
        }
    }

}

//--------------------------------------------------
//说明：计算每个gs相交的体素个数，顺便记录一下每个gs范围的左下和右上坐标，核函数
//作者：Mr_Shi
//--------------------------------------------------
__global__ void touched_kernel(
        int n_gs, float vox_size,
        float x_min, float y_min, float z_min,
        const dim3 voxel_grid,
        const Eigen::Vector3f* gs_centers,
        const Eigen::Vector3f* gs_rotvecs,
        const Eigen::Vector3f* gs_scales,
        Eigen::Vector3f* gs_lbs,
        Eigen::Vector3f* gs_rus,
        int* touched)
{
    auto idx = cg::this_grid().thread_rank();   // 每个gaussian为一个线程
    if (idx >= n_gs)
        return;
    Eigen::Vector3f center = gs_centers[idx];
    Eigen::Vector3f rotvec = gs_rotvecs[idx];
    Eigen::Vector3f scale = gs_scales[idx]*N_SIGMA;   // 取3*scale大小的范围

    Eigen::Vector3f box_vertex[8]={   // gs范围的八个顶点
        Eigen::Vector3f(-scale[0],-scale[1],-scale[2]), Eigen::Vector3f(-scale[0],-scale[1],scale[2]),
        Eigen::Vector3f(-scale[0],scale[1],-scale[2]), Eigen::Vector3f(-scale[0],scale[1],scale[2]),
        Eigen::Vector3f(scale[0],-scale[1],-scale[2]), Eigen::Vector3f(scale[0],-scale[1],scale[2]),
        Eigen::Vector3f(scale[0],scale[1],-scale[2]), Eigen::Vector3f(scale[0],scale[1],scale[2])
    };
    for(int i = 0; i < 8; i++)
    {
        Eigen::Vector3f pt = box_vertex[i];

        float norm = rotvec.norm();
        Eigen::Vector3f dir = rotvec.normalized();

        Eigen::AngleAxisf rotation_vector;
        if(norm < 1e-7) rotation_vector = Eigen::AngleAxisf::Identity();
        else rotation_vector = Eigen::AngleAxisf(norm, dir);

        box_vertex[i] = rotation_vector.matrix() * pt + center;
    }

    Eigen::Vector3f p_lb = box_vertex[0];
    Eigen::Vector3f p_ru = p_lb;
    for(int i = 1; i < 8; i++)  // 求八个顶点变换过后的最小值和最大值
    {
        p_lb = Eigen::Vector3f(
            min(p_lb[0], box_vertex[i][0]),
            min(p_lb[1], box_vertex[i][1]),
            min(p_lb[2], box_vertex[i][2])
        );
        p_ru = Eigen::Vector3f(
            max(p_ru[0], box_vertex[i][0]),
            max(p_ru[1], box_vertex[i][1]),
            max(p_ru[2], box_vertex[i][2])
        );
    }

    Eigen::Vector3f bounding_min(x_min,y_min,z_min);
    uint3 box_min, box_max;
    p_lb = p_lb-bounding_min;
    p_ru = p_ru-bounding_min;
    getBox(p_lb, p_ru, box_min, box_max, voxel_grid, vox_size);

    touched[idx] = (box_max.x-box_min.x)*(box_max.y-box_min.y)*(box_max.z-box_min.z);

    gs_lbs[idx] = p_lb;   // 同时保存每个gs在bounding范围局部坐标系中的左上和右下坐标，后面会用
    gs_rus[idx] = p_ru;

}

//--------------------------------------------------
//说明：根据体素ID与体素中心点与gs中心点的距离对局部SDF进行编码，保存在gs_keys_unsorted和gs_idx_unsorted中，核函数
//作者：Mr_Shi
//--------------------------------------------------
__global__ void encode_keys_by_voxel_kernel(int n_gs,
        float vox_size, dim3 voxel_grid,
        float x_min, float y_min, float z_min,
        const int* point_offsets,
        const Eigen::Vector3f* gs_centers,
        const Eigen::Vector3f* gs_rotvecs,
        const Eigen::Vector3f* gs_scales,
        const Eigen::Vector3f* gs_lbs,
        const Eigen::Vector3f* gs_rus,
        uint64_t* gs_keys_unsorted,
        int* gs_idx_unsorted)
{
    auto idx = cg::this_grid().thread_rank();   // 每个gaussian为一个线程
    if (idx >= n_gs)
        return;

    Eigen::Vector3f gs_center = gs_centers[idx];
    Eigen::Vector3f gs_rotvec = gs_rotvecs[idx];
    Eigen::Vector3f gs_scale = gs_scales[idx];

    Eigen::Vector3f p_lb = gs_lbs[idx];
    Eigen::Vector3f p_ru = gs_rus[idx];

    int offset = (idx == 0) ? 0 : point_offsets[idx - 1];
    uint3 box_min, box_max;
    getBox(p_lb, p_ru, box_min, box_max, voxel_grid, vox_size);

    Eigen::Vector3f bounding_lb(x_min, y_min, z_min);  //采样点范围的左下点坐标
    for(int z = box_min.z; z < box_max.z; z++)
    {
        for(int y = box_min.y; y < box_max.y; y++)
        {
            for(int x = box_min.x; x < box_max.x; x++)
            {
                Eigen::Vector3f vox_center(x+0.5f, y+0.5f, z+0.5f);
                vox_center = vox_center * vox_size + bounding_lb;   // 得到体素的中心点坐标

                float norm = gs_rotvec.norm();
                Eigen::Vector3f dir = gs_rotvec.normalized();
                Eigen::AngleAxisf rotation_vector;
                if(norm < 1e-7) rotation_vector = Eigen::AngleAxisf::Identity();
                else rotation_vector = Eigen::AngleAxisf(norm, dir);
                vox_center = rotation_vector.matrix().inverse() * (vox_center - gs_center);   // 旋转取逆

                vox_center = vox_center.array() / gs_scale.array();
                float dis = vox_center.norm();
                uint32_t dis_ = (uint32_t)(dis*1000000);

                uint64_t key = z*voxel_grid.x*voxel_grid.y + y*voxel_grid.x + x;   // 计算voxel的ID
                key <<= 32;      // 将voxel的ID放置到前32位
                key |= dis_;     // 将voxel中心与gaussian的距离放置到后32位
                gs_keys_unsorted[offset] = key;
                gs_idx_unsorted[offset] = idx;
                offset++;
            }
        }
    }

}

//--------------------------------------------------
//说明：确定在每个体素中的局部SDF的个数，核函数
//作者：Mr_Shi
//--------------------------------------------------
__global__ void identifyVoxelRanges_kernel(
        int num_rendered,
        const uint64_t* gs_keys,
        Eigen::Vector2i* ranges)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= num_rendered)
        return;
    uint64_t key = gs_keys[idx];
    uint32_t currvox = key >> 32;    // 从key中得到voxel_ID

    if(idx==0)
        ranges[currvox][0]=0;
    else
    {
        uint32_t prevvox = gs_keys[idx-1] >> 32;
        if(currvox != prevvox)
        {
            ranges[prevvox][1] = idx;
            ranges[currvox][0] = idx;
        }
    }
    if(idx == num_rendered-1)
        ranges[currvox][1] = num_rendered;
}

//--------------------------------------------------
//说明：查找距每个采样点最近的n_neibors个gs的idx，核函数
//作者：Mr_Shi
//--------------------------------------------------
__global__ void find_sp_neibers_kernel(
        int n_sp,
        int n_neibors,
        float vox_size,
        float x_min, float y_min, float z_min,
        dim3 voxel_grid,
        const Eigen::Vector3f* gs_centers,
        const Eigen::Vector3f* gs_rotvecs,
        const Eigen::Vector3f* gs_scales,
        const Eigen::Vector3f* sample_points,
        const Eigen::Vector2i* ranges,
        const int* gs_idx,
        float* neibors_dis2,
        int* neibors_idx)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= n_sp)
        return;

    Eigen::Vector3f sp = sample_points[idx];
    neibors_dis2 += (idx * n_neibors);
    neibors_idx += (idx * n_neibors);

    Eigen::Vector3f bounding_lb(x_min, y_min, z_min);  //采样点范围的左下点坐标
    Eigen::Vector3f sp_ = sp - bounding_lb;
    Eigen::Vector3i vox_coord((int)(sp_[0]/vox_size), (int)(sp_[1]/vox_size), (int)(sp_[2]/vox_size));
    int vox_id = vox_coord[2]*voxel_grid.x*voxel_grid.y + vox_coord[1]*voxel_grid.x + vox_coord[0];   // 计算voxel的ID

    Eigen::Vector2i range = ranges[vox_id];

    int k = 0;
    for(int i = range[0]; i < range[1]; i++)
    {
        if(k >= n_neibors)
            break;

        Eigen::Vector3f sp_t = sample_points[idx];

        Eigen::Vector3f gs_center = gs_centers[gs_idx[i]];
        Eigen::Vector3f gs_rotvec = gs_rotvecs[gs_idx[i]];
        Eigen::Vector3f gs_scale = gs_scales[gs_idx[i]];

        float norm = gs_rotvec.norm();
        Eigen::Vector3f dir = gs_rotvec.normalized();

        Eigen::AngleAxisf rotation_vector;
        if(norm < 1e-7) rotation_vector = Eigen::AngleAxisf::Identity();
        else rotation_vector = Eigen::AngleAxisf(norm, dir);
        sp_t = rotation_vector.matrix().inverse() * (sp_t - gs_center);   // 旋转取逆

        sp_t = sp_t.array() / gs_scale.array();
        float dis = sp_t.norm();

        if(dis <= 3.0f )   // 距离应该在标准高斯分布的3sigma范围内
        {
            neibors_idx[k] = gs_idx[i];
            neibors_dis2[k] = dis*dis;
        }

        k++;
    }


}

//--------------------------------------------------
//说明：计算gs和采样点的梯度，dim为输出特征的维度，核函数
//作者：Mr_Shi
//--------------------------------------------------
__global__ void compute_grad_kernel(
        int n_sp, int n_neibors,
        const Eigen::Vector3f* sample_points,
        const Eigen::Vector3f* gs_centers,
        const Eigen::Vector3f* gs_rotvecs,
        const Eigen::Vector3f* gs_scales,   // 已经过了exp函数
        const Eigen::Vector3f* neighb_vector_grad,
        const int* neiber_idx,
        float* sample_points_grad,
        float* gs_centers_grad,
        float* gs_rotvecs_grad,    // (dx,dy,dz)
        float* gs_scales_grad)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= n_sp)
        return;

    Eigen::Vector3f sp = sample_points[idx];
    neighb_vector_grad += (idx * n_neibors);
    neiber_idx += (idx * n_neibors);

    for(int i = 0; i < n_neibors; i++)
    {
        int neiber_id = neiber_idx[i];
        if(neiber_id < 0)
            continue;
        Eigen::Vector3f out_grad = neighb_vector_grad[i];
        Eigen::Vector3f gs_center = gs_centers[neiber_id];
        Eigen::Vector3f gs_rotvec = -gs_rotvecs[neiber_id];   // 取逆
        Eigen::Vector3f gs_scale = gs_scales[neiber_id];

        float theta = gs_rotvec.norm();
        float theta_half = 0.5f * theta, theta2 = theta * theta, theta4 = theta2 * theta2;
        Eigen::Quaternionf rot_quat;    // (w,x,y,z)
        if(theta < 1.2e-7)
        {
            float imag_factor = 0.5f - (1.0f/48.0f) * theta2 + (1.0f/3840.0f) * theta4;
            float real_factor = 1.0f - (1.0f/8.0f) * theta2 + (1.0f/384.0f) * theta4;
            rot_quat = Eigen::Quaternionf(real_factor, imag_factor*gs_rotvec[0], imag_factor*gs_rotvec[1], imag_factor*gs_rotvec[2]);
        }
        else
        {
            float imag_factor = sinf(theta_half) / theta;
            float real_factor = cosf(theta_half);
            rot_quat = Eigen::Quaternionf(real_factor, imag_factor*gs_rotvec[0], imag_factor*gs_rotvec[1], imag_factor*gs_rotvec[2]);
        }
        Eigen::Matrix3f rot_mat = rot_quat.matrix();

        // 求sample_points的导数
        Eigen::Vector3f sp_grad;
        Eigen::Vector3f p_t = out_grad.array()/gs_scale.array();
        sp_grad[0] = out_grad[0]*rot_mat(0,0)/gs_scale[0]+out_grad[1]*rot_mat(1,0)/gs_scale[1]+out_grad[2]*rot_mat(2,0)/gs_scale[2];
        sp_grad[1] = out_grad[0]*rot_mat(0,1)/gs_scale[0]+out_grad[1]*rot_mat(1,1)/gs_scale[1]+out_grad[2]*rot_mat(2,1)/gs_scale[2];
        sp_grad[2] = out_grad[0]*rot_mat(0,2)/gs_scale[0]+out_grad[1]*rot_mat(1,2)/gs_scale[1]+out_grad[2]*rot_mat(2,2)/gs_scale[2];
        for(int j = 0; j < 3; j++)
            atomicAdd(&sample_points_grad[idx*3+j], sp_grad[j]);
        // 求gs_centers的导数
        Eigen::Vector3f gs_center_grad;
        gs_center_grad[0] = -(out_grad[0]*rot_mat(0,0)/gs_scale[0]+out_grad[1]*rot_mat(1,0)/gs_scale[1]+out_grad[2]*rot_mat(2,0)/gs_scale[2]);
        gs_center_grad[1] = -(out_grad[0]*rot_mat(0,1)/gs_scale[0]+out_grad[1]*rot_mat(1,1)/gs_scale[1]+out_grad[2]*rot_mat(2,1)/gs_scale[2]);
        gs_center_grad[2] = -(out_grad[0]*rot_mat(0,2)/gs_scale[0]+out_grad[1]*rot_mat(1,2)/gs_scale[1]+out_grad[2]*rot_mat(2,2)/gs_scale[2]);
        for(int j = 0; j < 3; j++)
            atomicAdd(&gs_centers_grad[neiber_id*3+j], gs_center_grad[j]);
        // 求gs_scales的导数
        Eigen::Vector3f p_temp = rot_mat*(sp-gs_center);
        Eigen::Vector3f gs_scale_grad;
        gs_scale_grad[0] = -out_grad[0]*p_temp[0]/gs_scale[0];
        gs_scale_grad[1] = -out_grad[1]*p_temp[1]/gs_scale[1];
        gs_scale_grad[2] = -out_grad[2]*p_temp[2]/gs_scale[2];
        for(int j = 0; j < 3; j++)
            atomicAdd(&gs_scales_grad[neiber_id*3+j], gs_scale_grad[j]);
        // 求gs_rotvecs的导数
        Eigen::Vector3f gs_rotvec_grad;
        gs_rotvec_grad[0] = out_grad[0] / gs_scale[0];
        gs_rotvec_grad[1] = out_grad[1] / gs_scale[1];
        gs_rotvec_grad[2] = out_grad[2] / gs_scale[2];
        Eigen::Vector3f vec = rot_mat * (sp-gs_center);
        vec = -vec;
        Eigen::Matrix3f skew;
        skew<< 0.0f, -vec[2], vec[1],
               vec[2], 0.0f, -vec[0],
               -vec[1], vec[0], 0.0f;
        gs_rotvec_grad = skew.transpose() * gs_rotvec_grad;

        // 计算so3_Jl
        Eigen::Matrix3f K;
        K<< 0.0f, -gs_rotvec[2], gs_rotvec[1],
               gs_rotvec[2], 0.0f, -gs_rotvec[0],
               -gs_rotvec[1], gs_rotvec[0], 0.0f;
        float coef1, coef2;
        if(theta > 1.2e-7)
        {
            coef1 = (1 - cosf(theta)) / theta2;
            coef2 = (theta - sinf(theta)) / (theta * theta2);
        }
        else
        {
            coef1 =  0.5f - (1.0f/24.0f) * theta2;
            coef2 = 1.0f/6.0f - (1.0f/120) * theta2;
        }
        Eigen::Matrix3f so3_Jl = Eigen::Matrix3f::Identity() + coef1 * K + coef2 * (K * K);

        gs_rotvec_grad = -so3_Jl.transpose() * gs_rotvec_grad;
        for(int j = 0; j < 3; j++)
            atomicAdd(&gs_rotvecs_grad[neiber_id*3+j], gs_rotvec_grad[j]);

    }

}

//--------------------------------------------------
//说明：计算gs的二阶梯度，核函数
//作者：Mr_Shi
//--------------------------------------------------
__global__ void compute_grad_grad_kernel(
    int n_sp, int n_neibors,
    const Eigen::Vector3f* d_query_points_grad,
    const Eigen::Vector3f* gs_rotvecs,
    const Eigen::Vector3f* gs_scales,
    const Eigen::Vector3f* neighb_vector_grad,
    const int* neiber_idx,
    float* gs_rotvecs_grad,
    float* gs_scales_grad,
    float* neighb_vector_grads_grad)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= n_sp)
        return;

    Eigen::Vector3f d_normal = d_query_points_grad[idx];
    neighb_vector_grad += (idx * n_neibors);
    neiber_idx += (idx * n_neibors);
    neighb_vector_grads_grad += (idx * n_neibors * 3);

    for(int i = 0; i < n_neibors; i++)
    {
        int neiber_id = neiber_idx[i];
        if(neiber_id < 0)
            continue;
        Eigen::Vector3f out_grad = neighb_vector_grad[i];
        Eigen::Vector3f gs_rotvec = -gs_rotvecs[neiber_id];  // 取逆
        Eigen::Vector3f gs_scale = gs_scales[neiber_id];

        float theta = gs_rotvec.norm();
        float theta_half = 0.5f * theta, theta2 = theta * theta, theta4 = theta2 * theta2;
        Eigen::Quaternionf rot_quat;    // (w,x,y,z)
        if(theta < 1.2e-7)
        {
            float imag_factor = 0.5f - (1.0f/48.0f) * theta2 + (1.0f/3840.0f) * theta4;
            float real_factor = 1.0f - (1.0f/8.0f) * theta2 + (1.0f/384.0f) * theta4;
            rot_quat = Eigen::Quaternionf(real_factor, imag_factor*gs_rotvec[0], imag_factor*gs_rotvec[1], imag_factor*gs_rotvec[2]);
        }
        else
        {
            float imag_factor = sinf(theta_half) / theta;
            float real_factor = cosf(theta_half);
            rot_quat = Eigen::Quaternionf(real_factor, imag_factor*gs_rotvec[0], imag_factor*gs_rotvec[1], imag_factor*gs_rotvec[2]);
        }
        Eigen::Matrix3f rot_mat = rot_quat.matrix();

        // 求gs_scales的导数
        Eigen::Vector3f p_temp = rot_mat * d_normal;
        Eigen::Vector3f gs_scale_grad;
        gs_scale_grad = -p_temp.array() * out_grad.array() / gs_scale.array();   
        for(int j = 0; j < 3; j++)
            atomicAdd(&gs_scales_grad[neiber_id*3+j], gs_scale_grad[j]);

        // 求neighb_vector_grad的导数
        Eigen::Vector3f neighb_vector_grad_grad;
        neighb_vector_grad_grad = (rot_mat * d_normal).array() / gs_scale.array();  
        for(int j = 0; j < 3; j++)
            atomicAdd(&neighb_vector_grads_grad[i*3+j], neighb_vector_grad_grad[j]);

        // 求gs_rotvecs的导数
        Eigen::Vector3f gs_rotvec_grad;
        p_temp = out_grad.array() / gs_scale.array();

        Eigen::Matrix3f d_R = d_normal * p_temp.transpose();

        Eigen::Vector3f d_q;
        float x = rot_quat.x(), y = rot_quat.y(), z = rot_quat.z(), w = rot_quat.w();
        d_q[0] = -4*x*(d_R(1,1)+d_R(2,2))+2*y*(d_R(0,1)+d_R(1,0))+2*z*(d_R(0,2)+d_R(2,0))+2*w*(d_R(2,1)-d_R(1,2));
        d_q[1] = 2*x*(d_R(0,1)+d_R(1,0))-4*y*(d_R(0,0)+d_R(2,2))+2*z*(d_R(1,2)+d_R(2,1))+2*w*(d_R(0,2)-d_R(2,0));
        d_q[2] = 2*x*(d_R(2,0)+d_R(0,2))+2*y*(d_R(1,2)+d_R(2,1))-4*z*(d_R(0,0)+d_R(1,1))+2*w*(d_R(1,0)-d_R(0,1));
        // 计算so3_Jl
        Eigen::Matrix3f K;
        K<< 0.0f, -gs_rotvec[2], gs_rotvec[1],
            gs_rotvec[2], 0.0f, -gs_rotvec[0],
            -gs_rotvec[1], gs_rotvec[0], 0.0f;
        float coef1, coef2;
        if(theta > 1.2e-7)
        {
            coef1 = (1 - cosf(theta)) / theta2;
            coef2 = (theta - sinf(theta)) / (theta * theta2);
        }
        else
        {
            coef1 =  0.5f - (1.0f/24.0f) * theta2;
            coef2 = 1.0f/6.0f - (1.0f/120) * theta2;
        }
        Eigen::Matrix3f so3_Jl = Eigen::Matrix3f::Identity() + coef1 * K + coef2 * (K * K);  

        gs_rotvec_grad = -so3_Jl.transpose() * d_q;

        for(int j = 0; j < 3; j++)
            atomicAdd(&gs_rotvecs_grad[neiber_id*3+j], gs_rotvec_grad[j]);

    }
}

// =========================================================================

//--------------------------------------------------
//说明：根据给定的矩形边界，查找与矩形相交的gaussian，返回掩码,cuda函数
//作者：Mr_Shi
//--------------------------------------------------
void find_gs_by_boundingbox_cuda(int n_gs, float x_min, float y_min, float z_min, float x_max, float y_max, float z_max,
                const float* gs_centers, const float* gs_rotvecs, const float* gs_scales, bool* inside_mask)
{
    find_gs_by_boundingbox_kernel<<<(n_gs+255)/256, 256>>>(
        n_gs, x_min, y_min, z_min,
        x_max, y_max, z_max,
        (Eigen::Vector3f*) gs_centers,
        (Eigen::Vector3f*) gs_rotvecs,
        (Eigen::Vector3f*) gs_scales,
        inside_mask);
}

//--------------------------------------------------
//说明：计算每个gs相交的体素个数，保存在touched中,顺便记录一下每个gs范围的左下和右上坐标，cuda函数
//作者：Mr_Shi
//--------------------------------------------------
size_t touched_cuda(int n_gs, float vox_size, float x_min, float y_min, float z_min, int grid_x, int grid_y, int grid_z,
               const float* gs_centers, const float* gs_rotvecs, const float* gs_scales, float* gs_lbs, float* gs_rus, int* touched)
{
    dim3 voxel_grid(grid_x, grid_y, grid_z);

    touched_kernel<<<(n_gs+255)/256, 256>>>(
        n_gs, vox_size,
        x_min, y_min, z_min,
        voxel_grid,
        (Eigen::Vector3f*) gs_centers,
        (Eigen::Vector3f*) gs_rotvecs,
        (Eigen::Vector3f*) gs_scales,
        (Eigen::Vector3f*) gs_lbs,
        (Eigen::Vector3f*) gs_rus,
        touched);

    size_t scan_size;
    cub::DeviceScan::InclusiveSum(nullptr, scan_size, touched, touched, n_gs);

    return scan_size;
}

//--------------------------------------------------
//说明：对touched进行累加，保存在point_offsets中,cuda函数
//作者：Mr_Shi
//--------------------------------------------------
int offsets_cuda(int n_gs, size_t scan_size, int* touched, char* scanning_space, int* point_offsets)
{
    // 对touched累加，保存在point_offsets中
    cub::DeviceScan::InclusiveSum(scanning_space, scan_size, touched, point_offsets, n_gs);
    int num_rendered;
    cudaMemcpy(&num_rendered, point_offsets + n_gs - 1, sizeof(int), cudaMemcpyDeviceToHost);

    return num_rendered;
}

//--------------------------------------------------
//说明：根据体素ID与体素中心点与gs中心点的距离对局部SDF进行编码，保存在gs_keys_unsorted和gs_idx_unsorted中,cuda函数
//作者：Mr_Shi
//--------------------------------------------------
size_t encode_keys_by_voxel_cuda(int n_gs, int num_rendered, float vox_size, int grid_x, int grid_y, int grid_z, float x_min, float y_min, float z_min,
                    const int* point_offsets, const float* gs_centers, const float* gs_rotvecs, const float* gs_scales,
                    const float* gs_lbs, const float* gs_rus, long* gs_keys_unsorted, int* gs_idx_unsorted)
{
    dim3 voxel_grid(grid_x, grid_y, grid_z);

    encode_keys_by_voxel_kernel<<<(n_gs+255)/256, 256>>>(
         n_gs, vox_size,
         voxel_grid, x_min, y_min, z_min,
         point_offsets,
         (Eigen::Vector3f*) gs_centers,
         (Eigen::Vector3f*) gs_rotvecs,
         (Eigen::Vector3f*) gs_scales,
         (Eigen::Vector3f*) gs_lbs,
         (Eigen::Vector3f*) gs_rus,
         (uint64_t*) gs_keys_unsorted,
         gs_idx_unsorted);

    size_t sorting_size;
    cub::DeviceRadixSort::SortPairs(nullptr, sorting_size, (uint64_t*)gs_keys_unsorted, (uint64_t*)gs_keys_unsorted,
            gs_idx_unsorted, gs_idx_unsorted, num_rendered);

    return sorting_size;

}

//--------------------------------------------------
//说明：根据key进行排序，先按体素ID排，再按体素中心点到gs中心点的距离排,
//...并确定在每个体素中的局部SDF的个数（保存在ranges中），cuda函数
//作者：Mr_Shi
//--------------------------------------------------
void sort_keys_ranges_cuda(int num_rendered, size_t sorting_size, const long* gs_keys_unsorted, const int* gs_idx_unsorted,
                    char* sorting_space, long* gs_keys, int* gs_idx, int* ranges)
{
    cub::DeviceRadixSort::SortPairs(sorting_space, sorting_size, (uint64_t*)gs_keys_unsorted, (uint64_t*)gs_keys,
            gs_idx_unsorted, gs_idx, num_rendered);

    identifyVoxelRanges_kernel<< <(num_rendered + 255) / 256, 256 >> >(num_rendered, (uint64_t*)gs_keys, (Eigen::Vector2i*) ranges);
}

//--------------------------------------------------
//说明：查找距每个采样点最近的n_neibors个gs的idx，cuda函数
//作者：Mr_Shi
//--------------------------------------------------
void find_sp_neibers_cuda(int n_sp, int n_neibors, int grid_x, int grid_y, int grid_z, float vox_size, float x_min, float y_min, float z_min,
                          const float* gs_centers, const float* gs_rotvecs, const float* gs_scales, const float* sample_points, const int* ranges,
                          const int* gs_idx, float* neibors_dis2, int* neibors_idx)
{
    dim3 voxel_grid(grid_x, grid_y, grid_z);

    find_sp_neibers_kernel<<<(n_sp + 255) / 256, 256 >>>(
        n_sp, n_neibors, vox_size,
        x_min, y_min, z_min, voxel_grid,
        (Eigen::Vector3f*) gs_centers,
        (Eigen::Vector3f*) gs_rotvecs,
        (Eigen::Vector3f*) gs_scales,
        (Eigen::Vector3f*) sample_points,
        (Eigen::Vector2i*) ranges,
        gs_idx,
        neibors_dis2,
        neibors_idx);

    CUDA_CHECK_ERRORS();
}

//--------------------------------------------------
//说明：计算gs和采样点的梯度，dim为输出特征的维度，cuda函数
//作者：Mr_Shi
//--------------------------------------------------
void compute_grad_cuda(int n_sp, int n_neibors, const float* sample_points, const float* gs_centers, const float* gs_rotvecs, const float* gs_scales, const float* neighb_vector_grad, const int* neiber_idx,
                  float* sample_points_grad, float* gs_centers_grad, float* gs_quats_grad, float* gs_scales_grad)
{
    compute_grad_kernel<<<(n_sp + 255) / 256, 256 >>>(
        n_sp, n_neibors,
        (Eigen::Vector3f*) sample_points,
        (Eigen::Vector3f*) gs_centers,
        (Eigen::Vector3f*) gs_rotvecs,
        (Eigen::Vector3f*) gs_scales,
        (Eigen::Vector3f*) neighb_vector_grad,
        neiber_idx,
        sample_points_grad,
        gs_centers_grad,
        gs_quats_grad,
        gs_scales_grad);
}

//--------------------------------------------------
//说明：计算gs的二阶梯度，cuda函数
//作者：Mr_Shi
//--------------------------------------------------
void compute_grad_grad_cuda(int n_sp, int n_neibors, const float* d_query_points_grad, const float* gs_rotvecs, const float* gs_scales, const float* neighb_vector_grad, const int* neiber_idx,
                        float* gs_rotvecs_grad, float* gs_scales_grad, float* neighb_vector_grads_grad)
{
    compute_grad_grad_kernel<<<(n_sp + 255) / 256, 256 >>>(
        n_sp, n_neibors,
        (Eigen::Vector3f*) d_query_points_grad,    // (n_sp, 3)
        (Eigen::Vector3f*) gs_rotvecs,               // (n_gs, 3)
        (Eigen::Vector3f*) gs_scales,              // (n_gs, 3)
        (Eigen::Vector3f*) neighb_vector_grad,           // (n_sp, n_neibors, 3)
        neiber_idx,                     // (n_sp, n_neibors)
        gs_rotvecs_grad,                       // (n_gs, 3)
        gs_scales_grad,                      // (n_gs, 3)
        neighb_vector_grads_grad);           // (n_sp, n_neibors, 3)
}




