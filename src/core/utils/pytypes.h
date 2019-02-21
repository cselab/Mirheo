#pragma once 

#include <tuple>
#include <array>
#include <vector>
#include <cuda_runtime.h>

// Tuple initializations

namespace PyTypes
{

using float2 = std::tuple<float, float>;
using float3 = std::tuple<float, float, float>;
using float4 = std::tuple<float, float, float, float>;
    
using int2 = std::tuple<int, int>;
using int3 = std::tuple<int, int, int>;

template <class T, int Dimensions>
using VectorOfTypeN = std::vector<std::array<T, Dimensions>>;

template <int Dimensions>
using VectorOfFloatN = VectorOfTypeN<float, Dimensions>;
    
using VectorOfFloat  = VectorOfFloatN<1>;
using VectorOfFloat2 = VectorOfFloatN<2>;
using VectorOfFloat3 = VectorOfFloatN<3>;
using VectorOfFloat4 = VectorOfFloatN<4>;
using VectorOfFloat5 = VectorOfFloatN<5>;
using VectorOfFloat6 = VectorOfFloatN<6>;
using VectorOfFloat7 = VectorOfFloatN<7>;
using VectorOfFloat8 = VectorOfFloatN<8>;

template <int Dimensions>
using VectorOfIntN = VectorOfTypeN<int, Dimensions>;

using VectorOfInt  = VectorOfIntN<1>;
using VectorOfInt2 = VectorOfIntN<2>;
using VectorOfInt3 = VectorOfIntN<3>;
using VectorOfInt4 = VectorOfIntN<4>;

} // namespace PyTypes

inline float2 make_float2(PyTypes::float2 t2)
{
    return make_float2(std::get<0>(t2), std::get<1>(t2));
}
inline float3 make_float3(PyTypes::float3 t3)
{
    return make_float3(std::get<0>(t3), std::get<1>(t3), std::get<2>(t3));
}
inline float4 make_float4(PyTypes::float4 t4)
{
    return make_float4(std::get<0>(t4), std::get<1>(t4), std::get<2>(t4), std::get<3>(t4));
}

inline int2 make_int2(PyTypes::int2 t2)
{
    return make_int2(std::get<0>(t2), std::get<1>(t2));
}
inline int3 make_int3(PyTypes::int3 t3)
{
    return make_int3(std::get<0>(t3), std::get<1>(t3), std::get<2>(t3));
}
