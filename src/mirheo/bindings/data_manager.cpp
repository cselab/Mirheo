// Copyright 2021 ETH Zurich. All Rights Reserved.

#include "bindings.h"
#include <mirheo/core/pvs/data_manager.h>
#include <mirheo/core/utils/strprintf.h>

namespace mirheo
{

using namespace py::literals;

/** Returns '<' if little-endian is used and '>' otherwise.

    https://numpy.org/doc/stable/reference/arrays.interface.html#object.__array_interface__
*/
static char getEndiannessNumpy()
{
    uint16_t x = (uint16_t)'<' | (uint16_t)((uint16_t)'>' << 16);
    char out[2];
    memcpy(out, &x, 2);
    return out[0];
}

namespace
{

struct NumpyPartialInfo
{
    int secondDimShape = -1;
    int elementSize = -1;
    char type;
};

template <typename T>
struct GetNumpyPartialInfo
{
    static NumpyPartialInfo get()
    {
        throw std::runtime_error("Accessing this channel type not implemented.");
    }
};

#define MIR_NUMPY_PARTIAL_INFO(T, _2ndShape, size, type)  \
    template <>                                           \
    struct GetNumpyPartialInfo<T>                         \
    {                                                     \
        constexpr static NumpyPartialInfo get()           \
        {                                                 \
            return {(_2ndShape), (size), (type)};         \
        }                                                 \
    }

// Cupy does not support custom dtypes, so everything has to be exported as
// zero or multidimensional primitive types. It is not clear how to export
// Stress, RigidMotion and COMandExtent, so we skip disable it for now.
MIR_NUMPY_PARTIAL_INFO(int, -1, sizeof(int), 'd');
MIR_NUMPY_PARTIAL_INFO(int64_t, -1, sizeof(int64_t), 'd');
MIR_NUMPY_PARTIAL_INFO(float, -1, sizeof(float), 'f');
MIR_NUMPY_PARTIAL_INFO(float2, 2, sizeof(float), 'f');
MIR_NUMPY_PARTIAL_INFO(float3, 3, sizeof(float), 'f');
MIR_NUMPY_PARTIAL_INFO(float4, 4, sizeof(float), 'f');
MIR_NUMPY_PARTIAL_INFO(double, -1, sizeof(double), 'f');
MIR_NUMPY_PARTIAL_INFO(double2, 2, sizeof(double), 'f');
MIR_NUMPY_PARTIAL_INFO(double3, 3, sizeof(double), 'f');
MIR_NUMPY_PARTIAL_INFO(double4, 4, sizeof(double), 'f');
// MIR_NUMPY_PARTIAL_INFO(Stress, 6, sizeof(real), 'f');
// MIR_NUMPY_PARTIAL_INFO(RigidMotion, 3 + 4 + 3 + 3 + 3 + 3, sizeof(real), 'f');
// MIR_NUMPY_PARTIAL_INFO(COMandExtent, 9, sizeof(real), 'f');
MIR_NUMPY_PARTIAL_INFO(Force, 3, sizeof(real), 'f');  // <-- The int part is skipped!

struct NumpyArrayInfo
{
    int ndim;
    int shape[2];
    int strides[2];
    std::string type;
};

} // anonymous namespace


static NumpyArrayInfo getArrayInfo(DataManager::ChannelDescription &channel)
{
    return mpark::visit([&](auto *pinnedPtr) -> NumpyArrayInfo
    {
        using T = typename std::remove_reference_t<decltype(*pinnedPtr)>::value_type;
        auto partial = GetNumpyPartialInfo<T>::get();
        NumpyArrayInfo info;
        info.ndim = partial.secondDimShape > 0 ? 2 : 1;
        info.shape[0] = (int)pinnedPtr->size();
        info.shape[1] = partial.secondDimShape;
        info.strides[0] = (int)sizeof(T);
        info.strides[1] = partial.elementSize;

        // if (std::is_same<T, real4>::value && (
        //             channelName == channel_names::positions ||
        //             channelName == channel_names::velocities)) {
        //     // In the case of positions and velocities, we skip the 4th element.
        //     assert(partial.shape[1] == 4);
        //     partial.shape[1] = 3;
        // }
        info.type = strprintf("%c%c%d", getEndiannessNumpy(),
                              partial.type, partial.elementSize);
        return info;
    }, channel.varDataPtr);
}

/// Convert an int array to a Python tuple.
static py::tuple toTuple(const int *array, int size)
{
    py::tuple out(size);
    for (int i = 0; i < size; ++i)
        out[i] = array[i];
    return out;
}

static py::dict channelCudaArrayInterface(DataManager::ChannelDescription& channel)
{
    const bool readOnly = false;
    NumpyArrayInfo info = getArrayInfo(channel);
    return py::dict(
            "shape"_a = toTuple(info.shape, info.ndim),
            "typestr"_a = std::move(info.type),
            "data"_a = py::make_tuple((uintptr_t)channel.container->genericDevPtr(), readOnly),
            "version"_a = 2,
            "strides"_a = toTuple(info.strides, info.ndim));
}


void exportDataManagerChannel(py::module& m)
{
    using ChannelDescription = DataManager::ChannelDescription;

    py::class_<ChannelDescription> pycd(m, "ChannelDescription", R"(
        Storage for a single particle vector channel.
    )");

    pycd.def("__len__", [](const ChannelDescription *channel)
            {
                return channel->container->size();
            }, R"(
                The number of particles.
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
