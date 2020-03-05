#pragma once 

#include <mirheo/core/datatypes.h>

#include <tuple>
#include <array>
#include <vector>
#include <cuda_runtime.h>

namespace mirheo
{

// Tuple initializations

namespace py_types
{

using real2 = std::tuple<real, real>;
using real3 = std::tuple<real, real, real>;
using real4 = std::tuple<real, real, real, real>;
    
using int2 = std::tuple<int, int>;
using int3 = std::tuple<int, int, int>;

template <class T, int Dimensions>
using VectorOfTypeN = std::vector<std::array<T, Dimensions>>;

template <int Dimensions>
using VectorOfRealN = VectorOfTypeN<real, Dimensions>;
    
using VectorOfReal  = VectorOfRealN<1>;
using VectorOfReal2 = VectorOfRealN<2>;
using VectorOfReal3 = VectorOfRealN<3>;
using VectorOfReal4 = VectorOfRealN<4>;
using VectorOfReal5 = VectorOfRealN<5>;
using VectorOfReal6 = VectorOfRealN<6>;
using VectorOfReal7 = VectorOfRealN<7>;
using VectorOfReal8 = VectorOfRealN<8>;

template <int Dimensions>
using VectorOfIntN = VectorOfTypeN<int, Dimensions>;

using VectorOfInt  = VectorOfIntN<1>;
using VectorOfInt2 = VectorOfIntN<2>;
using VectorOfInt3 = VectorOfIntN<3>;
using VectorOfInt4 = VectorOfIntN<4>;

inline mirheo::real2 make_real2(real2 t2)
{
    return {std::get<0>(t2), std::get<1>(t2)};
}
inline mirheo::real3 make_real3(real3 t3)
{
    return {std::get<0>(t3), std::get<1>(t3), std::get<2>(t3)};
}
inline mirheo::real4 make_real4(real4 t4)
{
    return {std::get<0>(t4), std::get<1>(t4), std::get<2>(t4), std::get<3>(t4)};
}

inline ::int2 make_int2(int2 t2)
{
    return {std::get<0>(t2), std::get<1>(t2)};
}
inline ::int3 make_int3(int3 t3)
{
    return {std::get<0>(t3), std::get<1>(t3), std::get<2>(t3)};
}

} // namespace py_types
} // namespace mirheo
