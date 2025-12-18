#include "utils.h"
#include "cuda_utils.h"

//DDA光线相交测试，b为block_num=256，m为thread_num=M/block_num（M为光线数）
void dda_ray_intersect_cuda(int b, int m, int ht_array_size, int ht_size, float voxelsize, int n_max,
                       const float* ray_start, const float* ray_dir, const float* dminmax, const int* ht_info,
                       int* idx, float* min_depth, float* max_depth);

void inverse_cdf_sampling_cuda(int b, int num_rays, int max_hits, int max_steps, float fixed_step_size,
    const int *pts_idx, const float *min_depth, const float *max_depth, const float *uniform_noise,
    const float *probs, const float *steps, int *sampled_idx, float *sampled_depth, float *sampled_dists);
