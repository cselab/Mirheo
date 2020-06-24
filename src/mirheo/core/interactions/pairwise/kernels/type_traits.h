// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "density.h"
#include "mdpd.h"
#include "sdpd.h"

namespace mirheo
{

// interface

/** \brief A type trait that states if a density kernel needs to count the self interaction
    \tparam T The density kernel type

    By default, all density kernels need self interaction.
    Please add a template specialization if it is not the case.
 */
template <typename T>
struct needSelfInteraction
{
    /// default type trait value, must be overwritten by specialized cases
    static constexpr bool value = false;
};

/** \brief A type trait that states if a pairwise kernel outputs a force
    \tparam T The kernel type

    By default, all kernels do output a force.
    Please add a template specialization if it is not the case.
 */
template <class T>
struct outputsForce
{
    /// default type trait value, must be overwritten by specialized cases
    static constexpr bool value = true;
};

/** \brief A type trait that states if a pairwise kernel outputs density
    \tparam T The kernel type

    By default, all kernels do not output density.
    Please add a template specialization if it is the case.
 */
template <class T>
struct outputsDensity
{
    /// default type trait value, must be overwritten by specialized cases
    static constexpr bool value = !outputsForce<T>::value;
};


/** \brief A type trait that states if a pairwise kernel needs densities as input.
    \tparam T The kernel type

    By default, kernels do not need density as input.
    Please add a template specialization if it is the case.
 */
template <class T>
struct requiresDensity
{
    /// default type trait value, must be overwritten by specialized cases
    static constexpr bool value = false;
};

/** \brief A type trait that states if a pairwise kernel is of type Final.
    \tparam T The kernel type
 */
template <class T>
struct isFinal
{
    /// default type trait value, must be overwritten by specialized cases
    static constexpr bool value = outputsForce<T>::value;
};

// specialization
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe

template <>
struct needSelfInteraction<WendlandC2DensityKernel>
{ static constexpr bool value = true; };

template <typename T>
struct needSelfInteraction<PairwiseDensity<T>>
{ static constexpr bool value = needSelfInteraction<T>::value; };


template <class T>
struct outputsForce<PairwiseDensity<T>>
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

#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace mirheo
