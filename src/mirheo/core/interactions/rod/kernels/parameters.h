#pragma once

#include <mirheo/core/datatypes.h>
#include <random>

namespace mirheo
{

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

struct StatesParametersNone {};

struct StatesSmoothingParameters
{
    real kSmoothing;
};

struct StatesSpinParameters
{
    int nsteps;
    real kBT;
    real J;

    inline auto generate() {return udistr_(gen_);}
    
private:

    std::mt19937 gen_;
    std::uniform_real_distribution<real> udistr_;
};

} // namespace mirheo
