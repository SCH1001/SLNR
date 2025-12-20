cmake_minimum_required(VERSION 3.0)

project(sparse_hash)

# Use modern C++
set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_PREFIX_PATH /media/rvg/Disk2/anaconda3/envs/slnr/lib/python3.8/site-packages/torch/share/cmake)
find_package(Torch REQUIRED)
if (TORCH_INCLUDE_DIRS)
    include_directories("${TORCH_INCLUDE_DIRS}")
endif (TORCH_INCLUDE_DIRS)

# Eigen3
find_package(Eigen3 3.4 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
endif (EIGEN3_INCLUDE_DIR)


find_package(CUDA 11.8 REQUIRED)
if (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++14 -gencode arch=compute_89,code=sm_89 --expt-relaxed-constexpr)
endif ()
# ------------------------------------------------

cuda_add_library(svh_cuda SHARED svh_cuda.cu)
target_link_libraries(svh_cuda ${TORCH_LIBRARIES})

add_library(svh SHARED svh.cpp bindings.cpp)
target_link_libraries(svh svh_cuda ${TORCH_LIBRARIES})

