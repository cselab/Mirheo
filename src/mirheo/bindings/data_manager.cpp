// Copyright 2021 ETH Zurich. All Rights Reserved.

#include "bindings.h"
#include "cuda_array_interface.h"

namespace mirheo
{

using ChannelDescription = DataManager::ChannelDescription;

static py::dict channelCudaArrayInterface(ChannelDescription &channel)
{
    return getVariantCudaArrayInterface(channel.varDataPtr).cudaArrayInterface();
}

void exportDataManagerChannel(py::module& m)
{
    py::class_<ChannelDescription> pycd(m, "ChannelDescription", R"(
        Storage for a single particle vector channel.
    )");

    pycd.def("__len__", [](const ChannelDescription *channel)
            {
                return channel->container->size();
            }, R"(
                The number of elements.
            )");

    pycd.def_property_readonly("__cuda_array_interface__", &channelCudaArrayInterface, R"(
        The dictionary describing the underlying CUDA buffer.

        For more information, see:
            https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
        and
            https://docs.cupy.dev/en/stable/reference/interoperability.html#numba
    )");
}

} // namespace mirheo
