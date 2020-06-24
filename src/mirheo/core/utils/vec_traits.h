// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <vector_types.h>

namespace mirheo
{

namespace vec_traits
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

} // namespace vec_traits

} // namespace mirheo
