#pragma once

/// Structure keeping elastic parameters of the RBC model
struct MembraneParameters
{
    float x0, ks, ka, kd, kv, gammaC, gammaT, kbT, mpow, totArea0, totVolume0;
    bool fluctuationForces;
};

struct KantorBendingParameters
{
    float kb, theta;
};

struct JuelicherBendingParameters
{
    float kb, C0, kad, DA0;
};
