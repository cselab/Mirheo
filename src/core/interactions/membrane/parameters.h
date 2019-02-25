#pragma once

/// Structure keeping common parameters of the RBC model
struct MembraneParameters
{
    float x0, ks, ka, kd, kv, gammaC, gammaT, kbT, mpow, totArea0, totVolume0;
    bool fluctuationForces;
};

/// structure containing WLC bond + local area energy parameters
struct WLCParameters
{
    float x0, ks, mpow; ///< bond parameters
    float kd;           ///< local area energy
    float totArea0;     ///< equilibrium totalarea (not used for stress free case, used to compute eq length and local areas)
};

/// structure containing Lim shear energy parameters
struct LimParameters
{
    float ka;
    float a3, a4;
    float mu;
    float b1, b2;
    float totArea0;     ///< equilibrium totalarea (not used for stress free case, used to compute eq length and local areas)
};

/// structure containing Kanto bending parameters
struct KantorBendingParameters
{
    float kb, theta;
};

/// structure containing Juelicher bending + ADE parameters
struct JuelicherBendingParameters
{
    float kb, C0, kad, DA0;
};
