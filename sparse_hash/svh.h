#include <vector>
#include <memory>
#include <torch/script.h>
#include <torch/custom_class.h>
using namespace std;

#define INF 999999

//点云中的点
struct CC_Point
{
    float x;
    float y;
    float z;
};

class Voxel : public torch::CustomClassHolder
{
public:
    inline Voxel(int64_t vox_x, int64_t vox_y, int64_t vox_z)  //参数为体素坐标，为整型
    {
        _coord[0] = vox_x; _coord[1] = vox_y; _coord[2] = vox_z;
        _type = 0;
        _next_voxel = nullptr;
    }

    int _coord[3];   //体素坐标（左下角坐标），为整型
    char _type;      //哈希冲突类型，0为不冲突，1为冲突
    vector<float> _local_pointcloud;   //网格中的局部点云，按照哈希体素的遍历顺序对应存储
    vector<float> _local_pc_normal;
    Voxel* _next_voxel;
};

//使用链表法解决哈希冲突
class HashTable : public torch::CustomClassHolder
{
public:
    HashTable(double voxel_size, int64_t ht_size);
    ~HashTable();

    void insert(torch::Tensor pts);
    void delete_voxel(int64_t n_connect_th);
    std::tuple<at::Tensor, at::Tensor, at::Tensor> get_ht_info();
    std::tuple<at::Tensor, at::Tensor, at::Tensor> ray_intersect(at::Tensor ray_start, at::Tensor ray_dir, at::Tensor dminmax, at::Tensor ht_info, const int64_t n_max);
    std::tuple<at::Tensor, at::Tensor, at::Tensor> inverse_cdf_sampling(
        at::Tensor pts_idx, at::Tensor min_depth, at::Tensor max_depth, at::Tensor uniform_noise,
        at::Tensor probs, at::Tensor steps, double fixed_step_size);


    float _voxel_size;   //体素的尺寸大小
    int _ht_size;        //哈希表的空间大小
    int _vox_num;        //哈希表中体素的个数
    int _vox_conflict_num;        //哈希表中冲突的体素个数
    int _local_pointcloud_num;    //点云中点的个数
    vector<Voxel*> _vox_array;
};
