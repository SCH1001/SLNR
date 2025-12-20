#include "gs_search.h"

TORCH_LIBRARY(gs_search, m)
{
    m.class_<GS_Search>("GS_Search")
        .def(torch::init<>())
        .def("find_gs_by_boundingbox", &GS_Search::find_gs_by_boundingbox)
        .def("find_neibors", &GS_Search::find_neibors)
        .def("compute_grad", &GS_Search::compute_grad)
        .def("compute_grad_grad", &GS_Search::compute_grad_grad);

}
