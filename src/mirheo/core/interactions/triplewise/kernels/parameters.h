// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>

#include <variant>

namespace mirheo
{

// forward declaration of triplewise kernels
class SW3;
class TriplewiseDummy;

/// Stillinger-Weber (three-body term) parameters
struct SW3Params
{
    using KernelType = SW3; ///< the corresponding kernel
    real lambda;  ///< strength of the three-body term
    real epsilon; ///< energy scale
    real theta;   ///< equilibrium angle
    real gamma;   ///< decay length scale of the angular term
    real sigma;   ///< length scale
};

/// parameters of the dummy interaction
struct DummyParams
{
    using KernelType = TriplewiseDummy; ///< the corresponding kernel
    real epsilon;   ///< force coefficient
};

/// variant of all possible triplewise interactions
using VarTriplewiseParams = std::variant<SW3Params, DummyParams>;

} // namespace mirheo
