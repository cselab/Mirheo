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
    float x0, ks, mpow;
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
