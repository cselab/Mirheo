#pragma once 

#include <tuple>
#include <array>
#include <vector>
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
using pyfloat4 = std::tuple<float, float, float, float>;
using pyint3 = std::tuple<int, int, int>;

inline float3 make_float3(pyfloat3 t3)
{
    return make_float3(std::get<0>(t3), std::get<1>(t3), std::get<2>(t3));
}

inline float4 make_float4(pyfloat4 t4)
{
    return make_float4(std::get<0>(t4), std::get<1>(t4), std::get<2>(t4), std::get<3>(t4));
}

inline int3 make_int3(pyint3 t3)
{
    return make_int3(std::get<0>(t3), std::get<1>(t3), std::get<2>(t3));
}


using ICvector = std::vector<std::array<float, 7>>;
using PyContainer = std::vector<std::array<float, 3>>;
