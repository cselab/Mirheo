#include "cuda_array_interface.h"
#include <mirheo/core/utils/strprintf.h>
#include <mirheo/core/pvs/data_manager.h>

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

/// Convert an int array to a Python tuple.
static py::tuple toTuple(const int *array, int size)
{
    py::tuple out(size);
    for (int i = 0; i < size; ++i)
        out[i] = array[i];
    return out;
}

py::dict CudaArrayInterface::cudaArrayInterface() const
{
    assert(ndim <= 2);
    const bool isReadOnly = false;
    return py::dict(
            "shape"_a = toTuple(shape, ndim),
            "typestr"_a = type,
            "data"_a = py::make_tuple((uintptr_t)ptr, isReadOnly),
            "version"_a = 2,
            "strides"_a = toTuple(strides, ndim));
}


namespace
{

struct PartialArrayInterface
{
    int secondDimShape = -1;
    int elementSize = -1;
    char type;
};

template <typename T>
struct GetPartialArrayInterface
{
    static PartialArrayInterface get()
    {
        throw std::runtime_error("Accessing this channel type not implemented.");
    }
};

#define MIR_NUMPY_PARTIAL_INFO(T, _2ndShape, size, type)  \
    template <>                                           \
    struct GetPartialArrayInterface<T>                    \
    {                                                     \
        constexpr static PartialArrayInterface get()      \
        {                                                 \
            return {(_2ndShape), (int)(size), (type)};    \
        }                                                 \
    }

// Cupy does not support structured dtypes, so it's best to export everything
// using primitive types only (numba doesn't seem to have such limitation).
// Optional structured bindings could be added in the future, for example as
// `pv.local[channel_name].structured`.
MIR_NUMPY_PARTIAL_INFO(int, -1, sizeof(int), 'i');
MIR_NUMPY_PARTIAL_INFO(int64_t, -1, sizeof(int64_t), 'i');
MIR_NUMPY_PARTIAL_INFO(float, -1, sizeof(float), 'f');
MIR_NUMPY_PARTIAL_INFO(float2, 2, sizeof(float), 'f');
MIR_NUMPY_PARTIAL_INFO(float3, 3, sizeof(float), 'f');
MIR_NUMPY_PARTIAL_INFO(float4, 4, sizeof(float), 'f');
MIR_NUMPY_PARTIAL_INFO(double, -1, sizeof(double), 'f');
MIR_NUMPY_PARTIAL_INFO(double2, 2, sizeof(double), 'f');
MIR_NUMPY_PARTIAL_INFO(double3, 3, sizeof(double), 'f');
MIR_NUMPY_PARTIAL_INFO(double4, 4, sizeof(double), 'f');
MIR_NUMPY_PARTIAL_INFO(Stress, 6, sizeof(real), 'f');
// RigidMotion has padding in the middle due to alignment of the quaternion.
// Disabling for now, since exporting it in a naive fashion would expose the
// dummy element which would be quite impractical.
// MIR_NUMPY_PARTIAL_INFO(RigidMotion, 3 + 4 + 3 + 3 + 3 + 3, sizeof(real), 'f');
MIR_NUMPY_PARTIAL_INFO(COMandExtent, 9, sizeof(real), 'f');
MIR_NUMPY_PARTIAL_INFO(Force, 3, sizeof(real), 'f');  // <-- The int part is skipped!

} // anonymous namespace

static CudaArrayInterface toCudaArrayInterface(
        PartialArrayInterface partial, size_t size, size_t elementSize, void *ptr)
{
    CudaArrayInterface info;
    info.ndim = partial.secondDimShape > 0 ? 2 : 1;
    info.shape[0] = (int)size;
    info.shape[1] = partial.secondDimShape;
    info.strides[0] = elementSize;
    info.strides[1] = partial.elementSize;
    info.type = strprintf("%c%c%d", getEndiannessNumpy(),
                          partial.type, partial.elementSize);
    info.ptr = ptr;
    return info;
}

template <typename T>
CudaArrayInterface getBufferCudaArrayInterface(PinnedBuffer<T>& buffer)
{
    return toCudaArrayInterface(
            GetPartialArrayInterface<T>::get(),
            buffer.size(), sizeof(T), buffer.genericDevPtr());
}

// Explicitly instantiate the template with all types, because the template
// function is needed from other files as well.
#define MIR_WRAPPER(T) template CudaArrayInterface getBufferCudaArrayInterface<T>(PinnedBuffer<T>&);
MIRHEO_TYPE_TABLE(MIR_WRAPPER)
#undef MIR_WRAPPER

CudaArrayInterface getVariantCudaArrayInterface(VarPinnedBufferPtr& bufferVariant)
{
    return std::visit([](auto *pinnedBuffer) -> CudaArrayInterface
    {
        return getBufferCudaArrayInterface(*pinnedBuffer);
    }, bufferVariant);
};


void exportCudaArrayInterface(py::module& m)
{
    py::class_<CudaArrayInterface> pycai(m, "CudaArrayInterface", R"(
        Cupy and numba-compatible view for an internal CUDA buffer.
    )");

    pycai.def_property_readonly("__cuda_array_interface__", &CudaArrayInterface::cudaArrayInterface, R"(
        The dictionary describing the underlying CUDA buffer.

        For more information, see:
            https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
        and
            https://docs.cupy.dev/en/stable/reference/interoperability.html#numba
    )");
}

} // namespace mirheo
