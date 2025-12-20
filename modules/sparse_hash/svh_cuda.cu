#include "svh_cuda.h"
#include <cfloat>
#include <Eigen/Eigen>
#include <thrust/extrema.h>

#define INF 999999

//AABB相交检测函数
__device__ float2 RayAABBIntersection(
    const float3 &ori,
    const float3 &dir,
    const float3 &center,
    float half_voxel)
{

  float f_low = 0;
  float f_high = 100000.;
  float f_dim_low, f_dim_high, temp, inv_ray_dir, start, aabb;

  for (int d = 0; d < 3; ++d)
  {
    switch (d)
    {
    case 0:
      inv_ray_dir = __fdividef(1.0f, dir.x);
      start = ori.x;
      aabb = center.x;
      break;
    case 1:
      inv_ray_dir = __fdividef(1.0f, dir.y);
      start = ori.y;
      aabb = center.y;
      break;
    case 2:
      inv_ray_dir = __fdividef(1.0f, dir.z);
      start = ori.z;
      aabb = center.z;
      break;
    }

    f_dim_low = (aabb - half_voxel - start) * inv_ray_dir;
    f_dim_high = (aabb + half_voxel - start) * inv_ray_dir;

    if (f_dim_high < f_dim_low)
    {
      temp = f_dim_low;
      f_dim_low = f_dim_high;
      f_dim_high = temp;
    }

    if (f_dim_high < f_low)
    {
      return make_float2(-1.0f, -1.0f);
    }

    if (f_dim_low > f_high)
    {
      return make_float2(-1.0f, -1.0f);
    }

    f_low = (f_dim_low > f_low) ? f_dim_low : f_low;
    f_high = (f_dim_high < f_high) ? f_dim_high : f_high;

    if (f_low > f_high)
    {
      return make_float2(-1.0f, -1.0f);
    }
  }
  return make_float2(f_low, f_high);
}

//------------------------------------------------------------------------
//说明：在哈希数组中进行查询
//作者：Mr_Shi
//------------------------------------------------------------------------
__device__ bool find_voxel_ht(const int* ht_info, int voxel_x, int voxel_y, int voxel_z, int ht_array_size, int ht_size, int& hash_index)
{
    //计算哈希索引
    typedef unsigned int  uint;
    int index = (((uint)voxel_x * 73856093u) ^ ((uint)voxel_y * 19349669u) ^ ((uint)voxel_z * 83492791u)) & (uint)(ht_size-1);

    while(ht_info[index*4] != INF)
    {
        if(voxel_x == ht_info[index*4] && voxel_y == ht_info[index*4+1] && voxel_z == ht_info[index*4+2])
        {
            hash_index = index;
            return true;
        }
        else
        {
            index = ht_info[index*4+3];
            if(index == -1)
            {
                hash_index = -1;
                return false;
            }
            if(index >= ht_array_size)
            {
                hash_index = -1;
                printf("Hash table query out of bounds!\n");
                return false;
            }
        }
    }
    hash_index = -1;
    return false;

}

//------------------------------------------------------------------------
//说明：DDA光线相交核函数，b为block_num=256，n为哈希表数组的长度，m为thread_num=M/block_num（M为光线数）（b个grid，m个thread）
//问题：在ht_info中不会发生访问冲突么？答：访问冲突就顺序执行
//作者：Mr_Shi
//------------------------------------------------------------------------
__global__ void dda_intersect_point_kernel(
    int b, int m, int n, int ht_size, float voxelsize,
    int n_max,
    const float *__restrict__ ray_start,  //[256,M/256,3]，M为光线的个数
    const float *__restrict__ ray_dir,    //[256,M/256,3]
    const float *__restrict__ dminmax,    //[256,M/256,2]
    const int *__restrict__ ht_info,      //[256,N,4]，N为哈希数组的长度
    int *__restrict__ idx,
    float *__restrict__ min_depth,
    float *__restrict__ max_depth)
{
    int batch_index = blockIdx.x;         //共256，每个grid处理256批光线中的一批，每批是M/256条光线
    ht_info += batch_index * n * 4;
    ray_start += batch_index * m * 3;
    ray_dir += batch_index * m * 3;
    dminmax += batch_index * m * 2;
    idx += batch_index * m * n_max;
    min_depth += batch_index * m * n_max;
    max_depth += batch_index * m * n_max;

    int index = threadIdx.x;     //每个线程处理第batch_index批次的其中一个或多个光线
    int stride = blockDim.x;    //指的是一个grid有几个线程
    float half_voxel = voxelsize * 0.5;

    for (int j = index; j < m; j += stride)    //index为某一条光线的索引，每个线程处理第batch_index批次的其中一个或多个光线
    {
        for (int l = 0; l < n_max; ++l)
          idx[j * n_max + l] = -1;   //相交索引初始化

        Eigen::Vector3f ray_o(ray_start[j*3+0], ray_start[j*3+1], ray_start[j*3+2]);    //该条光向的原点
        Eigen::Vector3f ray_d(ray_dir[j*3+0], ray_dir[j*3+1], ray_dir[j*3+2]);          //该条光线的方向（假设方向没有单位化）
        float dmin = dminmax[j*2+0], dmax = dminmax[j*2+1];
        //得到光线的起点和终点
        Eigen::Vector3f ray_s = ray_o + ray_d * dmin;
        Eigen::Vector3f ray_e = ray_o + ray_d * dmax;

        //3D_DDA
        int cnt = 0;  //计数，最大为n_max
        int hash_index;  //从find_voxel_ht函数获取到的哈希索引
        Eigen::Vector3i current_voxel(std::floor(ray_s[0]/voxelsize),
                                      std::floor(ray_s[1]/voxelsize),
                                      std::floor(ray_s[2]/voxelsize));

        //判断是否与起点处的voxel相交
        if(find_voxel_ht(ht_info, current_voxel[0], current_voxel[1], current_voxel[2], n, ht_size, hash_index))
        {
            float vox_center_x = current_voxel[0]*voxelsize+half_voxel;  //voxel的中心
            float vox_center_y = current_voxel[1]*voxelsize+half_voxel;
            float vox_center_z = current_voxel[2]*voxelsize+half_voxel;
            float2 depths = RayAABBIntersection(
                 make_float3(ray_o[0], ray_o[1], ray_o[2]),
                 make_float3(ray_d[0], ray_d[1], ray_d[2]),
                 make_float3(vox_center_x, vox_center_y, vox_center_z),
                 half_voxel);
            if (depths.x > -1.0f) //在python中计算的tminmax可能有些不对，导致DDA得到的voxel，不与光线相交，因此这里加个判断
            {
                idx[j * n_max + cnt] = hash_index;   //保存哈希索引
                //保存与hash_index处的voxel相交的最小深度与最大深度
                min_depth[j * n_max + cnt] = depths.x;
                max_depth[j * n_max + cnt] = depths.y;
                ++cnt;
                if(cnt >= n_max) return;
            }
        }

        Eigen::Vector3i last_voxel(std::floor(ray_e[0]/voxelsize),
                                   std::floor(ray_e[1]/voxelsize),
                                   std::floor(ray_e[2]/voxelsize));

        float stepX = (ray_d[0] >= 0) ? 1:-1;
        float stepY = (ray_d[1] >= 0) ? 1:-1;
        float stepZ = (ray_d[2] >= 0) ? 1:-1;

        float boundary_stepX = (ray_d[0] >= 0) ? 1:0;
        float boundary_stepY = (ray_d[1] >= 0) ? 1:0;
        float boundary_stepZ = (ray_d[2] >= 0) ? 1:0;

        float next_voxel_boundary_x = (current_voxel[0]+boundary_stepX)*voxelsize; // correct
        float next_voxel_boundary_y = (current_voxel[1]+boundary_stepY)*voxelsize; // correct
        float next_voxel_boundary_z = (current_voxel[2]+boundary_stepZ)*voxelsize; // correct

        float tMaxX = (ray_d[0]!=0) ? (next_voxel_boundary_x - ray_s[0])/ray_d[0] : DBL_MAX; //
        float tMaxY = (ray_d[1]!=0) ? (next_voxel_boundary_y - ray_s[1])/ray_d[1] : DBL_MAX; //
        float tMaxZ = (ray_d[2]!=0) ? (next_voxel_boundary_z - ray_s[2])/ray_d[2] : DBL_MAX; //

        float tDeltaX = (ray_d[0]!=0) ? voxelsize/ray_d[0]*stepX : DBL_MAX;
        float tDeltaY = (ray_d[1]!=0) ? voxelsize/ray_d[1]*stepY : DBL_MAX;
        float tDeltaZ = (ray_d[2]!=0) ? voxelsize/ray_d[2]*stepZ : DBL_MAX;

        int n_iters = 0;
        while(last_voxel != current_voxel && cnt < n_max && n_iters < 5000)
        {
            n_iters++;
            if (tMaxX < tMaxY)
            {
                if (tMaxX < tMaxZ)
                {
                    current_voxel[0] += stepX;
                    tMaxX += tDeltaX;
                }
                else
                {
                    current_voxel[2] += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }
            else
            {
                if (tMaxY < tMaxZ)
                {
                    current_voxel[1] += stepY;
                    tMaxY += tDeltaY;
                }
                else
                {
                    current_voxel[2] += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }
            if(find_voxel_ht(ht_info, current_voxel[0], current_voxel[1], current_voxel[2], n, ht_size, hash_index))
            {
                float vox_center_x = current_voxel[0]*voxelsize+half_voxel;  //voxel的中心
                float vox_center_y = current_voxel[1]*voxelsize+half_voxel;
                float vox_center_z = current_voxel[2]*voxelsize+half_voxel;
                float2 depths = RayAABBIntersection(
                     make_float3(ray_o[0], ray_o[1], ray_o[2]),
                     make_float3(ray_d[0], ray_d[1], ray_d[2]),
                     make_float3(vox_center_x, vox_center_y, vox_center_z),
                     half_voxel);
                if (depths.x > -1.0f)
                {
                    idx[j * n_max + cnt] = hash_index;   //保存哈希索引
                    //保存与hash_index处的voxel相交的最小深度与最大深度
                    min_depth[j * n_max + cnt] = depths.x;
                    max_depth[j * n_max + cnt] = depths.y;
                    ++cnt;
                }
            }
        }

    }
}


//------------------------------------------------------------------------
//说明：在光线上逆变换采样核函数
//作者：Mr_Shi
//------------------------------------------------------------------------
__global__ void inverse_cdf_sampling_kernel(
    int b, int num_rays,
    int max_hits,
    int max_steps,
    float fixed_step_size,
    const int *__restrict__ pts_idx,
    const float *__restrict__ min_depth,
    const float *__restrict__ max_depth,
    const float *__restrict__ uniform_noise,
    const float *__restrict__ probs,
    const float *__restrict__ steps,
    int *__restrict__ sampled_idx,
    float *__restrict__ sampled_depth,
    float *__restrict__ sampled_dists)
{

  int batch_index = blockIdx.x;
  int index = threadIdx.x;
  int stride = blockDim.x;

  pts_idx += batch_index * num_rays * max_hits;
  min_depth += batch_index * num_rays * max_hits;
  max_depth += batch_index * num_rays * max_hits;
  probs += batch_index * num_rays * max_hits;
  steps += batch_index * num_rays;

  uniform_noise += batch_index * num_rays * max_steps;
  sampled_idx += batch_index * num_rays * max_steps;
  sampled_depth += batch_index * num_rays * max_steps;
  sampled_dists += batch_index * num_rays * max_steps;

  for (int j = index; j < num_rays; j += stride)
  {
    int H = j * max_hits, K = j * max_steps;
    int curr_bin = 0, s = 0;

    float curr_min_depth = min_depth[H];
    float curr_max_depth = max_depth[H];
    float curr_min_cdf = 0;
    float curr_max_cdf = probs[H];
    float step_size = 1.0 / steps[j];
    float z_low = curr_min_depth;
    int total_steps = int(ceil(steps[j]));
    bool done = false;

    if (fixed_step_size > 0.0)
      step_size = fixed_step_size;

    for (int curr_step = 0; curr_step < total_steps; curr_step++)
    {
      float curr_cdf = (float(curr_step) + uniform_noise[K + curr_step]) * step_size;
      while (curr_cdf > curr_max_cdf)
      {
        sampled_idx[K + s] = pts_idx[H + curr_bin];
        sampled_dists[K + s] = (curr_max_depth - z_low);
        sampled_depth[K + s] = (curr_max_depth + z_low) * .5;

        curr_bin++;
        s++;
        if ((curr_bin >= max_hits) || (pts_idx[H + curr_bin] == -1))
        {
          done = true;
          break;
        }
        curr_min_depth = min_depth[H + curr_bin];
        curr_max_depth = max_depth[H + curr_bin];
        curr_min_cdf = curr_max_cdf;
        curr_max_cdf = curr_max_cdf + probs[H + curr_bin];
        z_low = curr_min_depth;
      }
      if (done)
        break;

      float u = (curr_cdf - curr_min_cdf) / (curr_max_cdf - curr_min_cdf);
      float z = curr_min_depth + u * (curr_max_depth - curr_min_depth);
      sampled_idx[K + s] = pts_idx[H + curr_bin];

      sampled_dists[K + s] = (z - z_low);
      sampled_depth[K + s] = (z + z_low) * .5;
      z_low = z;
      s++;
    }

    while ((z_low < curr_max_depth) && (~done) && (num_rays > (H + curr_bin)))
    {
      sampled_idx[K + s] = pts_idx[H + curr_bin];
      sampled_dists[K + s] = (curr_max_depth - z_low);
      sampled_depth[K + s] = (curr_max_depth + z_low) * .5;
      curr_bin++;
      s++;
      if ((curr_bin >= max_hits) || (pts_idx[curr_bin] == -1))
        break;

      curr_min_depth = min_depth[H + curr_bin];
      curr_max_depth = max_depth[H + curr_bin];
      z_low = curr_min_depth;
    }
  }
}


//------------------------------------------------------------------------
//说明：DDA光线相交并行算法，b为block_num=256，m为thread_num=M/block_num（M为光线数）（b个grid，m个thread）
//作者：Mr_Shi
//------------------------------------------------------------------------
void dda_ray_intersect_cuda(int b, int m, int ht_array_size, int ht_size, float voxelsize, int n_max,
                       const float* ray_start, const float* ray_dir, const float* dminmax, const int* ht_info,
                       int* idx, float* min_depth, float* max_depth)
{
    dda_intersect_point_kernel<<<b, opt_n_threads(m)>>>(    //核函数
      b, m, ht_array_size, ht_size, voxelsize, n_max, ray_start, ray_dir, dminmax, ht_info, idx, min_depth, max_depth);

    CUDA_CHECK_ERRORS();
}


//------------------------------------------------------------------------
//说明：在光线上逆变换采样并行算法
//作者：Mr_Shi
//------------------------------------------------------------------------
void inverse_cdf_sampling_cuda(int b, int num_rays, int max_hits, int max_steps, float fixed_step_size,
    const int *pts_idx, const float *min_depth, const float *max_depth, const float *uniform_noise,
    const float *probs, const float *steps, int *sampled_idx, float *sampled_depth, float *sampled_dists)
{
    inverse_cdf_sampling_kernel<<<b, opt_n_threads(num_rays)>>>(
          b, num_rays, max_hits, max_steps, fixed_step_size,
          pts_idx, min_depth, max_depth, uniform_noise, probs, steps,
          sampled_idx, sampled_depth, sampled_dists);

    CUDA_CHECK_ERRORS();
}


