#pragma once

enum class StressFreeState
{
    Active,
    Inactive
};

// predeclaration for convenience

template <StressFreeState stressFreeState> class TriangleWLCForce;
template <StressFreeState stressFreeState> class TriangleLimForce;

class DihedralKantor;
class DihedralJuelicher;

/// Structure keeping common parameters of the RBC model
struct CommonMembraneParameters
{
    float ka, kv, gammaC, gammaT, kBT, totArea0, totVolume0;
    bool fluctuationForces;
};

/// structure containing WLC bond + local area energy parameters
struct WLCParameters
{
    template <StressFreeState stressFreeState>
    using TriangleForce = TriangleWLCForce<stressFreeState>;
    
    float x0, ks, mpow; ///< bond parameters
    float kd;           ///< local area energy
    float totArea0;     ///< equilibrium totalarea (not used for stress free case, used to compute eq length and local areas)
};

/// structure containing Lim shear energy parameters
struct LimParameters
{
    template <StressFreeState stressFreeState>
    using TriangleForce = TriangleLimForce<stressFreeState>;

    float ka;
    float a3, a4;
    float mu;
    float b1, b2;
    float totArea0;     ///< equilibrium totalarea (not used for stress free case, used to compute eq length and local areas)
};

/// structure containing Kanto bending parameters
struct KantorBendingParameters
{
    using DihedralForce = DihedralKantor;
    float kb, theta;
};

/// structure containing Juelicher bending + ADE parameters
struct JuelicherBendingParameters
{
    using DihedralForce = DihedralJuelicher;
    float kb, C0, kad, DA0;
};
