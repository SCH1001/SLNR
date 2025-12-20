#include "utils.h"
#include "cuda_utils.h"

void find_gs_by_boundingbox_cuda(int n_gs, float x_min, float y_min, float z_min, float x_max, float y_max, float z_max,
                    const float* gs_centers, const float* gs_rotvecs, const float* gs_scales, bool* inside_mask);

size_t touched_cuda(int n_gs, float vox_size, float x_min, float y_min, float z_min, int grid_x, int grid_y, int grid_z,
                 const float* gs_centers, const float* gs_rotvecs, const float* gs_scales, float* gs_lbs, float* gs_rus, int* touched);

int offsets_cuda(int n_gs, size_t scan_size, int* touched, char* scanning_space, int* point_offsets);

size_t encode_keys_by_voxel_cuda(int n_gs, int num_rendered, float vox_size, int grid_x, int grid_y, int grid_z, float x_min, float y_min, float z_min,
                const int* point_offsets, const float* gs_centers, const float* gs_rotvecs, const float* gs_scales,
                const float* gs_lbs, const float* gs_rus, long* gs_keys_unsorted, int* gs_idx_unsorted);

void sort_keys_ranges_cuda(int num_rendered, size_t sorting_size, const long* gs_keys_unsorted, const int* gs_idx_unsorted,
                    char* sorting_space, long* gs_keys, int* gs_idx, int* ranges);

void find_sp_neibers_cuda(int n_sp, int n_neibors, int grid_x, int grid_y, int grid_z, float vox_size, float x_min, float y_min, float z_min,
                          const float* gs_centers, const float* gs_rotvecs, const float* gs_scales, const float* sample_points, const int* ranges,
                          const int* gs_idx, float* neibors_dis2, int* neibors_idx);

void compute_grad_cuda(int n_sp, int n_neibors, const float* sample_points, const float* gs_centers, const float* gs_rotvecs, const float* gs_scales, const float* neighb_vector_grad, const int* neiber_idx,
                  float* sample_points_grad, float* gs_centers_grad, float* gs_rotvecs_grad, float* gs_scales_grad);

void compute_grad_grad_cuda(int n_sp, int n_neibors, const float* d_query_points_grad, const float* gs_rotvecs, const float* gs_scales, const float* neighb_vector_grad, const int* neiber_idx,
                        float* gs_rotvecs_grad, float* gs_scales_grad, float* neighb_vector_grads_grad);

