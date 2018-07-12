#pragma once 

#include <tuple>
#include <cuda_runtime.h>

// Tuple initializations

using pyfloat3 = std::tuple<float, float, float>;
using pyint3 = std::tuple<int, int, int>;

inline float3 make_float3(pyfloat3 t3)
{
    return make_float3(std::get<0>(t3), std::get<1>(t3), std::get<2>(t3));
}

inline int3 make_int3(pyint3 t3)
{
    return make_int3(std::get<0>(t3), std::get<1>(t3), std::get<2>(t3));
}

