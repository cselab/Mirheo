#pragma once

struct RodParameters
{
    float kBending; ///< bending force magnitude
    float kTwist;   ///< twist force magnitude
    float tauEq;

    float a0;       ///< equilibrium length between two opposite material frame particles
    float l0;       ///< equilibrium length between two consecutive centerline particles
    float kBounds;  ///< bound force magnitude
};
