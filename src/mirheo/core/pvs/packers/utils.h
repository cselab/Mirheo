// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <cstddef>

namespace mirheo
{

template <typename TPadding = real4>
__HD__ constexpr static size_t getPaddedSize(size_t datumSize, int n)
{
    size_t size = n * datumSize;
    size_t npads = (size + sizeof(TPadding)-1) / sizeof(TPadding);
    return npads * sizeof(TPadding);
}

template <typename T, typename TPadding = real4>
__HD__ constexpr static size_t getPaddedSize(int n)
{
    return getPaddedSize<TPadding>(sizeof(T), n);
}

} // namespace mirheo
