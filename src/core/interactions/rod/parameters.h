#pragma once

struct RodParameters
{
    float3 kBending; ///< bending force magnitude in that order: (Bxx, Bxy, Byy) (symmetric matrix)
    float2 omegaEq;  ///< equilibrium curvature along the material frames

    float kTwist;   ///< twist force magnitude
    float tauEq;    ///< equilibrium torsion

    float a0;       ///< equilibrium length between two opposite material frame particles
    float l0;       ///< equilibrium length between two consecutive centerline particles
    float kBounds;  ///< bound force magnitude
    float kVisc;    ///< bound viscous force magnitude
};
