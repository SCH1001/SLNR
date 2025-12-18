#include <vector>
#include <memory>
#include <torch/script.h>
#include <torch/custom_class.h>
using namespace std;


class GS_Search : public torch::CustomClassHolder
{
public:
    GS_Search(){};
    ~GS_Search(){};

    at::Tensor find_gs_by_boundingbox(torch::Tensor gs_centers, torch::Tensor gs_rotvecs, torch::Tensor gs_scales, torch::Tensor bounding_min, torch::Tensor bounding_max);

    std::tuple<at::Tensor, at::Tensor> find_neibors(torch::Tensor sample_points, torch::Tensor gs_centers, torch::Tensor gs_rotvecs, torch::Tensor gs_scales, torch::Tensor bounding_min, torch::Tensor bounding_max, double vox_size, int64_t n_neibors);

    void compute_grad(torch::Tensor sample_points, torch::Tensor gs_centers, torch::Tensor gs_rotvecs, torch::Tensor gs_scales, torch::Tensor neighb_vector_grad, torch::Tensor neiber_idx,
                      torch::Tensor sample_points_grad, torch::Tensor gs_centers_grad, torch::Tensor gs_rotvecs_grad, torch::Tensor gs_scales_grad);

    void compute_grad_grad(torch::Tensor d_query_points_grad, torch::Tensor gs_rotvecs, torch::Tensor gs_scales, torch::Tensor neighb_vector_grad, torch::Tensor neiber_idx,
                           torch::Tensor gs_rotvecs_grad, torch::Tensor gs_scales_grad, torch::Tensor neighb_vector_grads_grad);

};
