#pragma once 

#include <mirheo/core/datatypes.h>

#include <array>
#include <vector>

namespace mirheo
{

namespace py_types
{

template <class T, int Dimensions>
using VectorOfTypeN = std::vector<std::array<T, Dimensions>>;

template <int Dimensions>
using VectorOfRealN = VectorOfTypeN<real, Dimensions>;
    
using VectorOfReal3 = VectorOfRealN<3>;

template <int Dimensions>
using VectorOfIntN = VectorOfTypeN<int, Dimensions>;

using VectorOfInt3 = VectorOfIntN<3>;


} // namespace py_types
} // namespace mirheo
