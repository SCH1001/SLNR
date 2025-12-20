#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x)                                             \
  do                                                              \
  {                                                               \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                            \
  do                                                                   \
  {                                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                                 \
  do                                                    \
  {                                                     \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
                #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                                 \
  do                                                      \
  {                                                       \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, \
                #x " must be a float tensor");            \
  } while (0)

#define CHECK_ERROR_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

