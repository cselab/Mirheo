// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <random>

namespace mirheo
{

/// parameters for computing rod forces
struct RodParameters
{
    real3 kBending; ///< bending force magnitude in that order: (Bxx, Bxy, Byy) (symmetric matrix)
    std::vector<real2> kappaEq;  ///< equilibrium curvature along the material frames (one per state)

    real kTwist;   ///< twist force magnitude
    std::vector<real> tauEq;    ///< equilibrium torsion (one per state)

    std::vector<real> groundE; ///< ground energy of each state

    real a0;        ///< equilibrium length between two opposite material frame particles
    real l0;        ///< equilibrium length between two consecutive centerline particles
    real ksCenter;  ///< spring force magnitude for centerline
    real ksFrame;   ///< spring force magnitude for material frame
};

/// Parameters used when no polymorphic states are considered
struct StatesParametersNone {};

/// Parameters used when polymorphic states transitions are modeled with smoothing energy
struct StatesSmoothingParameters
{
    real kSmoothing; ///< smoothing potential coefficient
};

/// Parameters used when polymorphic states transitions are modeled with Ising kind of model
struct StatesSpinParameters
{
    int nsteps; ///< Number of MC steps
    real kBT;   ///< temeperature in energy units
    real J;     ///< Ising energy coupling

    /// \return a random seed
    inline auto generate() {return udistr_(gen_);}

private:
    std::mt19937 gen_;
    std::uniform_real_distribution<real> udistr_;
};

} // namespace mirheo
