#include "svh.h"

TORCH_LIBRARY(svh, m)
{
    m.class_<Voxel>("Voxel")
        .def(torch::init<int64_t,int64_t,int64_t>());

    m.class_<HashTable>("HashTable")
        .def(torch::init<double,int64_t>())
        .def("insert", &HashTable::insert)
        .def("delete_voxel", &HashTable::delete_voxel)
        .def("get_ht_info", &HashTable::get_ht_info)
        .def("ray_intersect", &HashTable::ray_intersect)
        .def("inverse_cdf_sampling", &HashTable::inverse_cdf_sampling);

}
