#pragma once

#include "density.h"
#include "mdpd.h"
#include "sdpd.h"

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



template <class T>
struct outputsDensity
{
    static constexpr bool value = !outputsForce<T>::value;
};


template <class T>
struct requiresDensity
{
    static constexpr bool value = false;
};

template<>
struct requiresDensity<PairwiseMDPD>
{
    static constexpr bool value = true;
};

template <class T0, class T1>
struct requiresDensity<PairwiseSDPD<T0, T1>>
{
    static constexpr bool value = true;
};

