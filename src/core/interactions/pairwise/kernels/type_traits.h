#pragma once

#include "density.h"

template <typename T>
struct needSelfInteraction
{ static constexpr bool value = false; };

template <>
struct needSelfInteraction<WendlandC2DensityKernel>
{ static constexpr bool value = true; };

template <typename T>
struct needSelfInteraction<PairwiseDensity<T>>
{ static constexpr bool value = needSelfInteraction<T>::value; };



template <class T>
struct outputsForce
{
    static constexpr bool value = true;
};

template <class T>
struct outputsForce<PairwiseDensity<T>>
{
    static constexpr bool value = false;
};
