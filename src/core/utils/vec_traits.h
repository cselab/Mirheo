#pragma once

#include "vector_types.h"

namespace VecTraits
{

#define IMPLEMENT_VEC_N(type, N)                                        \
    template <> struct Vec <type, N> { using Type = type ## N; };

#define IMPLEMENT_VEC(type)                     \
    IMPLEMENT_VEC_N(type, 2)                    \
    IMPLEMENT_VEC_N(type, 3)                    \
    IMPLEMENT_VEC_N(type, 4)


template <typename T, int N> struct Vec {};

IMPLEMENT_VEC(float)
IMPLEMENT_VEC(double)
IMPLEMENT_VEC(int)

#undef IMPLEMENT_VEC
#undef IMPLEMENT_VEC_N

} // namespace VecTraits
