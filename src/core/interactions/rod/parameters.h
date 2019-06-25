#pragma once

struct RodParameters
{
    float3 kBending; ///< bending force magnitude in that order: (Bxx, Bxy, Byy) (symmetric matrix)
    std::vector<float2> kappaEq;  ///< equilibrium curvature along the material frames (one per state)

    float kTwist;   ///< twist force magnitude
    std::vector<float> tauEq;    ///< equilibrium torsion (one per state)

    std::vector<float> groundE; ///< ground energy of each state
    
    float a0;        ///< equilibrium length between two opposite material frame particles
    float l0;        ///< equilibrium length between two consecutive centerline particles
    float ksCenter;  ///< spring force magnitude for centerline
    float ksFrame;   ///< spring force magnitude for material frame
};

struct StatesPenalizationParameters
{
    // TODO; for now no penalization implemented
};

struct StatesSpinParameters
{
    int nsteps;
    float kBT;
    float J;
};
