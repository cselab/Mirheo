#pragma once 

#include <tuple>
#include <cuda_runtime.h>

// Tuple initializations

using pyfloat2 = std::tuple<float, float>;
using pyint2 = std::tuple<int, int>;

inline float2 make_float2(pyfloat2 t2)
{
    return make_float2(std::get<0>(t2), std::get<1>(t2));
}

inline int2 make_int2(pyint2 t2)
{
    return make_int2(std::get<0>(t2), std::get<1>(t2));
}



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

