#include "svh.h"
#include "svh_cuda.h"
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include <list>

using namespace std;

HashTable::HashTable(double voxel_size, int64_t ht_size)
{
    _voxel_size = voxel_size;
    _ht_size = ht_size;
    _vox_num = 0;
    _vox_conflict_num = 0;
    _local_pointcloud_num = 0;
    _vox_array.resize(_ht_size, nullptr);   //动态分配voxel数组

}

HashTable::~HashTable()
{
    for(int i = 0; i < _ht_size; i++)
    {
        Voxel* vox = _vox_array[i];
        Voxel* vox1 = nullptr;
        while (vox)
        {
            vox1 = vox->_next_voxel;
            delete vox;
            vox = vox1;
        }
    }
}

//--------------------------------------------------
//说明：哈希表插入操作，输入的是三维点坐标数组
//作者：Mr_Shi
//--------------------------------------------------
void HashTable::insert(torch::Tensor pts)
{
    auto points = pts.accessor<float, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        return;
    }

    for (int i = 0; i < points.size(0); i++)
    {
        int voxel_x = floor(points[i][0]/_voxel_size);
        int voxel_y = floor(points[i][1]/_voxel_size);
        int voxel_z = floor(points[i][2]/_voxel_size);

        //计算哈希索引
        typedef unsigned int  uint;
        uint index = (((uint)voxel_x * 73856093u) ^ ((uint)voxel_y * 19349669u) ^ ((uint)voxel_z * 83492791u)) & (uint)(_ht_size-1);
        //uint index = ((uint)voxel_x ^ ((uint)voxel_y * 2654435761) ^ ((uint)voxel_z * 805459861u)) & (uint)_ht_size;
        //查询
        Voxel* vox = _vox_array[index];
        Voxel* vox_before = nullptr;
        char c_conflict = 0;  //-1为voxel已存在，0为不发生冲突， 1为发生冲突
        while(vox)
        {
            //若哈希表中已存在该体素，则直接返回
            if(vox->_coord[0] == voxel_x && vox->_coord[1] == voxel_y && vox->_coord[2] == voxel_z)
            {
                c_conflict = -1;
                break;
            }
            vox_before = vox;
            vox = vox->_next_voxel;
            c_conflict = 1;   //确定发生哈希冲突
        }
        if(c_conflict == -1) continue;
        //插入
        Voxel* new_vox = new Voxel(voxel_x, voxel_y, voxel_z);
        if(c_conflict == 1)  //判断是否发生哈希冲突
        {
            new_vox->_type = 1;
            _vox_conflict_num++;   //哈希表中发生哈希冲突的体素个数加1
            vox_before->_next_voxel = new_vox;
        }
        else if(c_conflict == 0)
            _vox_array[index] = new_vox;
        _vox_num++;  //哈希表中体素个数加1
    }

}

//--------------------------------------------------
//说明：根据体素连通性与体素中局部点云的个数判断体素是否要删除
//...n_connect_th为连通个数阈值，若大于等于该阈值，则该体素的连通性检验通过
//作者：Mr_Shi
//--------------------------------------------------
void HashTable::delete_voxel(int64_t n_connect_th)
{
    struct XYZ {
        int x; int y; int z;
        bool operator<(const XYZ xyz) const{
            return (x<xyz.x) || (x==xyz.x && y<xyz.y) || (x==xyz.x && y==xyz.y && z<xyz.z);
        }
    };

    vector<int> lpc_count_array;    //记录当前体素中局部点云的个数
    vector<bool> connective_array;  //记录连通性
    //26邻域
    int offset_x[26] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1};
    int offset_y[26] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int offset_z[26] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1};

    //6邻域
//    int offset_x[6] = {-1, 1, 0, 0, 0, 0};
//    int offset_y[6] = {0, 0, -1, 1, 0, 0};
//    int offset_z[6] = {0, 0, 0, 0, -1, 1};

    for(int i = 0; i < _vox_array.size(); i++)
    {
        Voxel* vox = _vox_array[i];
        while(vox)
        {
            //1.记录当前体素中局部点云的个数
            lpc_count_array.push_back(vox->_local_pointcloud.size()/3);

            //2.检验连通性，记录是否通过连通性检验
            int n_connect = 0;   //当前体素连通的个数

            list<XYZ> xyz_list;
            set<XYZ> xyz_set;  //记录某个体素是否遍历过
            int voxel_x = vox->_coord[0], voxel_y = vox->_coord[1], voxel_z = vox->_coord[2];   //初始扩散体素坐标
            XYZ xyz = {voxel_x, voxel_y, voxel_z};
            xyz_list.push_back(xyz);
            while(!xyz_list.empty())
            {
                xyz = xyz_list.front();
                xyz_list.pop_front();
                xyz_set.insert(xyz);
                for(int k = 0; k < 26; k++)
                {
                    voxel_x = xyz.x + offset_x[k];
                    voxel_y = xyz.y + offset_y[k];
                    voxel_z = xyz.z + offset_z[k];
                    //计算哈希索引
                    typedef unsigned int  uint;
                    uint index = (((uint)voxel_x * 73856093u) ^ ((uint)voxel_y * 19349669u) ^ ((uint)voxel_z * 83492791u)) & (uint)(_ht_size-1);
                    Voxel* vox1 = _vox_array[index];  //查询
                    while(vox1)
                    {
                        XYZ xyz_temp = {voxel_x, voxel_y, voxel_z};
                        if(vox1->_coord[0] == voxel_x && vox1->_coord[1] == voxel_y && vox1->_coord[2] == voxel_z && xyz_set.find(xyz_temp) == xyz_set.end())
                        {
                            xyz_list.push_back(xyz_temp);   //若哈希表中存在该体素，则体素坐标进队，连通个数加1，停止查找
                            xyz_set.insert(xyz_temp);
                            n_connect++;
                            break;
                        }
                        vox1 = vox1->_next_voxel;
                    }
                }
                if(n_connect >= n_connect_th)  //若连通个数大于等于该阈值，则该体素的连通性检验通过，提前停止连通性检验
                {
                    connective_array.push_back(true);
                    break;
                }
            }
            if(n_connect < n_connect_th) //若连通个数小于该阈值，则该体素的连通性检验不通过
                connective_array.push_back(false);

            vox = vox->_next_voxel;
        }
    }
    //计算局部点云个数的阈值
    float mean_count = 1.0f*_local_pointcloud_num/_vox_num;
    float dev_sum = 0.0f;
    for(int i = 0; i < lpc_count_array.size(); i++)
    {
        float delt2 = abs(lpc_count_array[i]-mean_count);
        dev_sum += delt2*delt2;
    }
    float sigma = sqrt(dev_sum/_vox_num);
    float lcp_num_th = mean_count-3*sigma;    //得到局部点云个数的阈值
    lcp_num_th = mean_count/8;

    //再次遍历体素，删除局部点云个数过少的和连通性检验不通过的体素
    int N = 0;
    for(int i = 0; i < _vox_array.size(); i++)
    {
        Voxel* vox = _vox_array[i];
        Voxel* vox_before = nullptr;
        while(vox)
        {
            if(!connective_array[N]) //当前只进行连通性检验
            {
                _vox_num--;                   //哈希表中体素的个数减1
                _local_pointcloud_num -= lpc_count_array[N];    //点云中点的个数

                if(vox->_type == 1)
                {
                    vox_before->_next_voxel = vox->_next_voxel;
                    delete vox;      //删除该体素
                    vox = vox_before->_next_voxel;  //vox_before不变
                    _vox_conflict_num--;    //哈希表中冲突的体素个数减1
                }
                else
                {
                    _vox_array[i] = vox->_next_voxel;
                    delete vox;   //删除该体素
                    vox = _vox_array[i];
                    if(vox)
                    {
                        vox->_type = 0;
                        _vox_conflict_num--;    //哈希表中冲突的体素个数减1
                    }
                    vox_before = nullptr;
                }
            }
            else
            {
                vox_before = vox;
                vox = vox->_next_voxel;
            }
            N++;
        }
    }
}


//--------------------------------------------------
//说明：得到哈希表的信息（包括voxel坐标和索引，以及局部点云），以tuple形式返回
//作者：Mr_Shi
//--------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor> HashTable::get_ht_info()
{
    auto vox_x = torch::ones({_ht_size+_vox_conflict_num}, dtype(torch::kInt32)) * INF;
    auto vox_y = torch::ones({_ht_size+_vox_conflict_num}, dtype(torch::kInt32)) * INF;
    auto vox_z = torch::ones({_ht_size+_vox_conflict_num}, dtype(torch::kInt32)) * INF;
    auto next_index = -torch::ones({_ht_size+_vox_conflict_num}, dtype(torch::kInt32));   //-1表示无子节点
    //局部点云在点云数组中的起始索引和长度，-1表示该voxel中不包含局部点云
    auto local_pointcloud_id = -torch::ones({_ht_size+_vox_conflict_num}, dtype(torch::kInt32));
    //这里的len指的是点的个数，而不是该段数组的长度
    auto local_pointcloud_len = -torch::ones({_ht_size+_vox_conflict_num}, dtype(torch::kInt32));
    //点云数组
    vector<float> pointcloud_vec(_local_pointcloud_num*3);

    int excess_index = 0;

    int temp_id = 0;
    int temp_lenth = 0;
    for(int i = 0; i < _vox_array.size(); i++)
    {
        Voxel* vox = _vox_array[i];
        int j = i;
        while(vox)
        {
            vox_x[j] = vox->_coord[0];
            vox_y[j] = vox->_coord[1];
            vox_z[j] = vox->_coord[2];
            local_pointcloud_id[j] = temp_id/3;
            temp_lenth = vox->_local_pointcloud.size();
            copy(vox->_local_pointcloud.begin(), vox->_local_pointcloud.end(),
                 pointcloud_vec.begin()+temp_id);
            local_pointcloud_len[j] = temp_lenth/3;
            temp_id = temp_id + temp_lenth;   //更新temp_id

            if(vox->_next_voxel)
            {
                next_index[j] = _ht_size + excess_index;
                j = next_index[j].item<int>();   //下一个节点的位置
                excess_index++;
            }

            vox = vox->_next_voxel;
        }
    }
    auto pointcloud_array = torch::tensor(pointcloud_vec).reshape({-1, 3});

    auto ht_info = torch::cat({vox_x.reshape({vox_x.size(0), 1}), vox_y.reshape({vox_x.size(0), 1}),
                vox_z.reshape({vox_x.size(0), 1}), next_index.reshape({vox_x.size(0), 1})}, 1);
    auto local_pointcloud_index = torch::cat({local_pointcloud_id.reshape({vox_x.size(0), 1}),
                local_pointcloud_len.reshape({vox_x.size(0), 1})}, 1);

    return std::make_tuple(ht_info, local_pointcloud_index, pointcloud_array);
}


//--------------------------------------------------
//说明：光线相交，得到与光线相交的voxel（参数ray_dir为光线在世界坐标系的方向，没有归一化）
//作者：Mr_Shi
//--------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor> HashTable::ray_intersect(at::Tensor ray_start, at::Tensor ray_dir, at::Tensor dminmax, at::Tensor ht_info, const int64_t n_max)
{
    CHECK_CONTIGUOUS(ray_start); CHECK_CONTIGUOUS(ray_dir); CHECK_CONTIGUOUS(dminmax); CHECK_CONTIGUOUS(ht_info);
    CHECK_IS_FLOAT(ray_start); CHECK_IS_FLOAT(ray_dir); CHECK_IS_FLOAT(dminmax); CHECK_IS_INT(ht_info);
    CHECK_CUDA(ray_start); CHECK_CUDA(ray_dir); CHECK_CUDA(dminmax); CHECK_CUDA(ht_info);

    at::Tensor idx =
      torch::zeros({ray_start.size(0), ray_start.size(1), n_max},
                   at::device(ray_start.device()).dtype(at::ScalarType::Int));
    at::Tensor min_depth =
      torch::zeros({ray_start.size(0), ray_start.size(1), n_max},
                   at::device(ray_start.device()).dtype(at::ScalarType::Float));
    at::Tensor max_depth =
      torch::zeros({ray_start.size(0), ray_start.size(1), n_max},
                   at::device(ray_start.device()).dtype(at::ScalarType::Float));

    int block_num = ray_start.size(0), thread_num = ray_start.size(1), ht_array_size = ht_info.size(1);

    //cuda函数
    dda_ray_intersect_cuda(block_num, thread_num, ht_array_size, _ht_size, _voxel_size, n_max,
        ray_start.data_ptr<float>(), ray_dir.data_ptr<float>(), dminmax.data_ptr<float>(), ht_info.data_ptr<int>(),
        idx.data_ptr<int>(), min_depth.data_ptr<float>(), max_depth.data_ptr<float>());

    return std::make_tuple(idx, min_depth, max_depth);

}


//--------------------------------------------------
//说明：在光线上逆变换采样
//作者：Mr_Shi
//--------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor> HashTable::inverse_cdf_sampling(at::Tensor pts_idx, at::Tensor min_depth,
    at::Tensor max_depth, at::Tensor uniform_noise, at::Tensor probs, at::Tensor steps, double fixed_step_size)
{
    CHECK_CONTIGUOUS(pts_idx); CHECK_CONTIGUOUS(min_depth); CHECK_CONTIGUOUS(max_depth); CHECK_CONTIGUOUS(probs); CHECK_CONTIGUOUS(steps); CHECK_CONTIGUOUS(uniform_noise);
    CHECK_IS_INT(pts_idx); CHECK_IS_FLOAT(min_depth); CHECK_IS_FLOAT(max_depth); CHECK_IS_FLOAT(uniform_noise); CHECK_IS_FLOAT(probs); CHECK_IS_FLOAT(steps);
    CHECK_CUDA(pts_idx); CHECK_CUDA(min_depth); CHECK_CUDA(max_depth); CHECK_CUDA(uniform_noise); CHECK_CUDA(probs); CHECK_CUDA(steps);

    int max_steps = uniform_noise.size(-1);
    at::Tensor sampled_idx =
        -torch::ones({pts_idx.size(0), pts_idx.size(1), max_steps},
                     at::device(pts_idx.device()).dtype(at::ScalarType::Int));
    at::Tensor sampled_depth =
        torch::zeros({min_depth.size(0), min_depth.size(1), max_steps},
                     at::device(min_depth.device()).dtype(at::ScalarType::Float));
    at::Tensor sampled_dists =
        torch::zeros({min_depth.size(0), min_depth.size(1), max_steps},
                     at::device(min_depth.device()).dtype(at::ScalarType::Float));
    inverse_cdf_sampling_cuda(min_depth.size(0), min_depth.size(1), min_depth.size(2), sampled_depth.size(2), fixed_step_size, pts_idx.data_ptr<int>(),
        min_depth.data_ptr<float>(), max_depth.data_ptr<float>(), uniform_noise.data_ptr<float>(), probs.data_ptr<float>(),
        steps.data_ptr<float>(), sampled_idx.data_ptr<int>(), sampled_depth.data_ptr<float>(), sampled_dists.data_ptr<float>());
    return std::make_tuple(sampled_idx, sampled_depth, sampled_dists);
}





