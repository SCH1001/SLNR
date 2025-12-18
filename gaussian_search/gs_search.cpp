#include "gs_search.h"
#include "gs_search_cuda.h"
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include <list>

using namespace std;

//--------------------------------------------------
//说明：根据给定的矩形边界，查找与矩形相交的gaussian，返回掩码
//作者：Mr_Shi
//--------------------------------------------------
at::Tensor GS_Search::find_gs_by_boundingbox(torch::Tensor gs_centers, torch::Tensor gs_rotvecs, torch::Tensor gs_scales, torch::Tensor bounding_min, torch::Tensor bounding_max)
{
    CHECK_CONTIGUOUS(gs_centers); CHECK_CONTIGUOUS(gs_rotvecs); CHECK_CONTIGUOUS(gs_scales); CHECK_CONTIGUOUS(bounding_min); CHECK_CONTIGUOUS(bounding_max);
    CHECK_IS_FLOAT(gs_centers); CHECK_IS_FLOAT(gs_rotvecs); CHECK_IS_FLOAT(gs_scales); CHECK_IS_FLOAT(bounding_min); CHECK_IS_FLOAT(bounding_max);
    CHECK_CUDA(gs_centers); CHECK_CUDA(gs_rotvecs); CHECK_CUDA(gs_scales); CHECK_CUDA(bounding_min); CHECK_CUDA(bounding_max);

    int n_gs = gs_centers.size(0);
    float x_min = bounding_min[0].item<float>(), y_min = bounding_min[1].item<float>(), z_min = bounding_min[2].item<float>();
    float x_max = bounding_max[0].item<float>(), y_max = bounding_max[1].item<float>(), z_max = bounding_max[2].item<float>();
    at::Tensor inside_mask =torch::zeros({n_gs},at::device(gs_centers.device()).dtype(at::ScalarType::Bool));

    find_gs_by_boundingbox_cuda(n_gs, x_min, y_min, z_min, x_max, y_max, z_max,
        gs_centers.data_ptr<float>(), gs_rotvecs.data_ptr<float>(), gs_scales.data_ptr<float>(), inside_mask.data_ptr<bool>());

    return inside_mask;
}

//--------------------------------------------------
//说明：查询sample_points相交的gs，返回相交gs的idx
//作者：Mr_Shi
//--------------------------------------------------
std::tuple<at::Tensor, at::Tensor> GS_Search::find_neibors(torch::Tensor sample_points, torch::Tensor gs_centers, torch::Tensor gs_rotvecs, torch::Tensor gs_scales, torch::Tensor bounding_min, torch::Tensor bounding_max, double vox_size, int64_t n_neibors)
{
    CHECK_CONTIGUOUS(sample_points); CHECK_CONTIGUOUS(gs_centers); CHECK_CONTIGUOUS(gs_rotvecs); CHECK_CONTIGUOUS(gs_scales); CHECK_CONTIGUOUS(bounding_min); CHECK_CONTIGUOUS(bounding_max);
    CHECK_IS_FLOAT(sample_points); CHECK_IS_FLOAT(gs_centers); CHECK_IS_FLOAT(gs_rotvecs); CHECK_IS_FLOAT(gs_scales); CHECK_IS_FLOAT(bounding_min); CHECK_IS_FLOAT(bounding_max);
    CHECK_CUDA(sample_points); CHECK_CUDA(gs_centers); CHECK_CUDA(gs_rotvecs); CHECK_CUDA(gs_scales); CHECK_CUDA(bounding_min); CHECK_CUDA(bounding_max);

    int n_gs = gs_centers.size(0);
    float x_min = bounding_min[0].item<float>(), y_min = bounding_min[1].item<float>(), z_min = bounding_min[2].item<float>();
    float x_max = bounding_max[0].item<float>(), y_max = bounding_max[1].item<float>(), z_max = bounding_max[2].item<float>();
    float range_x = x_max - x_min;
    float range_y = y_max - y_min;
    float range_z = z_max - z_min;
    int grid_x = ceilf(range_x/vox_size);
    int grid_y = ceilf(range_y/vox_size);
    int grid_z = ceilf(range_z/vox_size);

    // 计算每个gs相交的体素个数，保存在touched中，顺便记录一下每个gs范围的左下和右上坐标
    at::Tensor touched =torch::zeros({n_gs},at::device(gs_centers.device()).dtype(at::ScalarType::Int));  //记录每个gs与多少个voxel相交
    at::Tensor gs_lbs = torch::empty({n_gs,3},at::device(gs_centers.device()).dtype(at::ScalarType::Float));
    at::Tensor gs_rus = torch::empty({n_gs,3},at::device(gs_centers.device()).dtype(at::ScalarType::Float));
    size_t scan_size = touched_cuda(n_gs, vox_size, x_min, y_min, z_min, grid_x, grid_y, grid_z, gs_centers.data_ptr<float>(),
              gs_rotvecs.data_ptr<float>(), gs_scales.data_ptr<float>(), gs_lbs.data_ptr<float>(), gs_rus.data_ptr<float>(), touched.data_ptr<int>());

    // 对touched进行累加，保存在point_offsets中
    at::Tensor point_offsets = torch::zeros({n_gs},at::device(gs_centers.device()).dtype(at::ScalarType::Int)); // 存储对touched累加值
    at::Tensor scanning_space = torch::empty({(long)scan_size},at::device(gs_centers.device()).dtype(at::ScalarType::Char)); //分配临时空间
    int num_rendered = offsets_cuda(n_gs, scan_size, touched.data_ptr<int>(), reinterpret_cast<char*>(scanning_space.data_ptr()), point_offsets.data_ptr<int>());

    // 根据体素ID与体素中心点与gs中心点的距离对局部SDF进行编码，保存在gs_keys_unsorted和gs_idx_unsorted中
    at::Tensor gs_keys_unsorted = torch::zeros({(long)num_rendered},at::device(gs_centers.device()).dtype(at::ScalarType::Long));
    at::Tensor gs_idx_unsorted = torch::zeros({(long)num_rendered},at::device(gs_centers.device()).dtype(at::ScalarType::Int));
    size_t sorting_size = encode_keys_by_voxel_cuda(n_gs, num_rendered, vox_size, grid_x, grid_y, grid_z, x_min, y_min, z_min,
                              point_offsets.data_ptr<int>(), gs_centers.data_ptr<float>(), gs_rotvecs.data_ptr<float>(), gs_scales.data_ptr<float>(),
                              gs_lbs.data_ptr<float>(), gs_rus.data_ptr<float>(), gs_keys_unsorted.data_ptr<long>(), gs_idx_unsorted.data_ptr<int>());

    // 对编码进行排序,先按体素ID排，再按体素中心点到gs中心点的距离排，并确定在每个体素中的局部SDF的个数（保存在ranges中）
    at::Tensor gs_keys = torch::zeros({(long)num_rendered},at::device(gs_centers.device()).dtype(at::ScalarType::Long));
    at::Tensor gs_idx = torch::zeros({(long)num_rendered},at::device(gs_centers.device()).dtype(at::ScalarType::Int));
    at::Tensor sorting_space = torch::empty({(long)sorting_size},at::device(gs_centers.device()).dtype(at::ScalarType::Char)); //分配临时空间
    at::Tensor ranges = torch::zeros({grid_x*grid_y*grid_z, 2},at::device(gs_centers.device()).dtype(at::ScalarType::Int));
    sort_keys_ranges_cuda(num_rendered, sorting_size, gs_keys_unsorted.data_ptr<long>(), gs_idx_unsorted.data_ptr<int>(),
                   reinterpret_cast<char*>(sorting_space.data_ptr()), gs_keys.data_ptr<long>(), gs_idx.data_ptr<int>(), ranges.data_ptr<int>());

    // 查询采样点的neibers
    int n_sp = sample_points.size(0);
    at::Tensor neibors_idx = -torch::ones({n_sp, n_neibors},at::device(gs_centers.device()).dtype(at::ScalarType::Int));
    at::Tensor neibors_dis2 = torch::ones({n_sp, n_neibors},at::device(gs_centers.device()).dtype(at::ScalarType::Float))*999999.0f;
    find_sp_neibers_cuda(n_sp, n_neibors, grid_x, grid_y, grid_z, vox_size, x_min, y_min, z_min,
                        gs_centers.data_ptr<float>(), gs_rotvecs.data_ptr<float>(), gs_scales.data_ptr<float>(),
                        sample_points.data_ptr<float>(), ranges.data_ptr<int>(), gs_idx.data_ptr<int>(),
                        neibors_dis2.data_ptr<float>(), neibors_idx.data_ptr<int>());


    return  std::make_tuple(neibors_dis2, neibors_idx);

}

//--------------------------------------------------
//说明：计算gs和采样点的梯度
//作者：Mr_Shi
//--------------------------------------------------
void GS_Search::compute_grad(torch::Tensor sample_points, torch::Tensor gs_centers, torch::Tensor gs_rotvecs, torch::Tensor gs_scales, torch::Tensor neighb_vector_grad, torch::Tensor neiber_idx,
                             torch::Tensor sample_points_grad, torch::Tensor gs_centers_grad, torch::Tensor gs_rotvecs_grad, torch::Tensor gs_scales_grad)
{
    CHECK_CONTIGUOUS(sample_points); CHECK_CONTIGUOUS(gs_centers); CHECK_CONTIGUOUS(gs_rotvecs); CHECK_CONTIGUOUS(gs_scales); CHECK_CONTIGUOUS(neighb_vector_grad); CHECK_CONTIGUOUS(neiber_idx); CHECK_CONTIGUOUS(sample_points_grad); CHECK_CONTIGUOUS(gs_centers_grad); CHECK_CONTIGUOUS(gs_rotvecs_grad); CHECK_CONTIGUOUS(gs_scales_grad);
    CHECK_IS_FLOAT(sample_points); CHECK_IS_FLOAT(gs_centers); CHECK_IS_FLOAT(gs_rotvecs); CHECK_IS_FLOAT(gs_scales); CHECK_IS_FLOAT(neighb_vector_grad); CHECK_IS_INT(neiber_idx); CHECK_IS_FLOAT(sample_points_grad); CHECK_IS_FLOAT(gs_centers_grad); CHECK_IS_FLOAT(gs_rotvecs_grad); CHECK_IS_FLOAT(gs_scales_grad);
    CHECK_CUDA(sample_points); CHECK_CUDA(gs_centers); CHECK_CUDA(gs_rotvecs); CHECK_CUDA(gs_scales); CHECK_CUDA(neighb_vector_grad); CHECK_CUDA(neiber_idx); CHECK_CUDA(sample_points_grad); CHECK_CUDA(gs_centers_grad); CHECK_CUDA(gs_rotvecs_grad); CHECK_CUDA(gs_scales_grad);

    int n_sp = neiber_idx.size(0);
    int n_neibors = neiber_idx.size(1);
    compute_grad_cuda(n_sp, n_neibors, sample_points.data_ptr<float>(), gs_centers.data_ptr<float>(), gs_rotvecs.data_ptr<float>(), gs_scales.data_ptr<float>(), neighb_vector_grad.data_ptr<float>(), neiber_idx.data_ptr<int>(),
                 sample_points_grad.data_ptr<float>(), gs_centers_grad.data_ptr<float>(), gs_rotvecs_grad.data_ptr<float>(), gs_scales_grad.data_ptr<float>());
}

//--------------------------------------------------
//说明：计算gs的二阶梯度
//作者：Mr_Shi
//--------------------------------------------------
void GS_Search::compute_grad_grad(torch::Tensor d_query_points_grad, torch::Tensor gs_rotvecs, torch::Tensor gs_scales, torch::Tensor neighb_vector_grad, torch::Tensor neiber_idx,
                       torch::Tensor gs_rotvecs_grad, torch::Tensor gs_scales_grad, torch::Tensor neighb_vector_grads_grad)
{
    CHECK_CONTIGUOUS(d_query_points_grad); CHECK_CONTIGUOUS(gs_rotvecs); CHECK_CONTIGUOUS(gs_scales); CHECK_CONTIGUOUS(neighb_vector_grad); CHECK_CONTIGUOUS(neiber_idx); CHECK_CONTIGUOUS(gs_rotvecs_grad); CHECK_CONTIGUOUS(gs_scales_grad); CHECK_CONTIGUOUS(neighb_vector_grads_grad);
    CHECK_IS_FLOAT(d_query_points_grad); CHECK_IS_FLOAT(gs_rotvecs); CHECK_IS_FLOAT(gs_scales); CHECK_IS_FLOAT(neighb_vector_grad); CHECK_IS_INT(neiber_idx); CHECK_IS_FLOAT(gs_rotvecs_grad); CHECK_IS_FLOAT(gs_scales_grad); CHECK_IS_FLOAT(neighb_vector_grads_grad);
    CHECK_CUDA(d_query_points_grad); CHECK_CUDA(gs_rotvecs); CHECK_CUDA(gs_scales); CHECK_CUDA(neighb_vector_grad); CHECK_CUDA(neiber_idx); CHECK_CUDA(gs_rotvecs_grad); CHECK_CUDA(gs_scales_grad); CHECK_CUDA(neighb_vector_grads_grad);

    int n_sp = neiber_idx.size(0);
    int n_neibors = neiber_idx.size(1);
    compute_grad_grad_cuda(n_sp, n_neibors, d_query_points_grad.data_ptr<float>(), gs_rotvecs.data_ptr<float>(), gs_scales.data_ptr<float>(), neighb_vector_grad.data_ptr<float>(), neiber_idx.data_ptr<int>(),
                 gs_rotvecs_grad.data_ptr<float>(), gs_scales_grad.data_ptr<float>(), neighb_vector_grads_grad.data_ptr<float>());

}







