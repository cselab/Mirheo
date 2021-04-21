// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/pvs/data_manager.h>

#include <pybind11/pybind11.h>

namespace mirheo {
namespace py = pybind11;

struct CudaArrayInterface
{
    int ndim;
    int shape[2];
    int strides[2];
    std::string type;
    void *ptr;

    /// Return the __cuda_array_interface__ dictionary for the underlying buffer.
    py::dict cudaArrayInterface() const;
};

/// Get a cupy/numba-compatible representation of the pinned buffer.
template <typename T>
CudaArrayInterface getBufferCudaArrayInterface(PinnedBuffer<T>& buffer);

/// Get a cupy/numba-compatible representation of the pinned buffer under the given variant.
CudaArrayInterface getVariantCudaArrayInterface(VarPinnedBufferPtr& bufferVariant);

void exportCudaArrayInterface(py::module& m);

} // namespace mirheo
