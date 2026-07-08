// Copyright 2021 ETH Zurich. All Rights Reserved.

#include "data_manager.h"
#include "cuda_array_interface.h"

#include <mirheo/core/pvs/data_manager.h>

namespace mirheo
{

using namespace py::literals;

void exportDataManager(py::module& m)
{
    py::class_<DataManager> pydm(m, "DataManager", R"(
        A collection of channels in pinned memory.
    )");

    pydm.def("__getitem__",
             [](DataManager *dm, const std::string &name)
             {
                 auto * const channel = dm->getChannelDesc(name);
                 if (channel == nullptr)
                     throw py::key_error(name);
                 return getVariantCudaArrayInterface(channel->varDataPtr);
             },
             "name"_a, py::keep_alive<0, 1>(),
             R"(
                 Returns:
                     Cupy-compatible view over the internal CUDA buffer.
             )");

    pydm.def("__iter__",
             [](DataManager *dm)
             {
                 const auto &channels = dm->getSortedChannels();
                 return py::make_key_iterator(channels.begin(), channels.end());
             },
             py::keep_alive<0, 1>(),
             R"(
                 Returns:
                     Iterator over channel names.
             )");
}

} // namespace mirheo
