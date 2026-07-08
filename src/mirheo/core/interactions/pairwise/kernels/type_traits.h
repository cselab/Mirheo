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

// specialization
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe

template <>
struct needSelfInteraction<WendlandC2DensityKernel>
{
    static constexpr bool value = true;
};

template <typename T>
struct needSelfInteraction<PairwiseDensity<T>>
{
    static constexpr bool value = needSelfInteraction<T>::value;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace mirheo
