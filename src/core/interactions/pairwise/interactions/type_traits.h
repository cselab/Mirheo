#pragma once

#include "density.h"

template <typename T>
struct needSelfInteraction
{ static const bool value = false; };

template <>
struct needSelfInteraction<WendlandC2DensityKernel>
{ static const bool value = true; };

template <typename T>
struct needSelfInteraction<PairwiseDensity<T>>
{ static const bool value = needSelfInteraction<T>::value; };
