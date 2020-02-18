#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

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
    real ka, kv, gammaC, gammaT, kBT, totArea0, totVolume0;
    bool fluctuationForces;
};
MIRHEO_MEMBER_VARS(CommonMembraneParameters, ka, kv, gammaC, gammaT, kBT,
                   totArea0, totVolume0, fluctuationForces);

/// structure containing WLC bond + local area energy parameters
struct WLCParameters
{
    template <StressFreeState stressFreeState>
    using TriangleForce = TriangleWLCForce<stressFreeState>;
    
    real x0, ks, mpow; ///< bond parameters
    real kd;           ///< local area energy
    real totArea0;     ///< equilibrium totalarea (not used for stress free case, used to compute eq length and local areas)
};
MIRHEO_MEMBER_VARS(WLCParameters, x0, ks, mpow, kd, totArea0);

/// structure containing Lim shear energy parameters
struct LimParameters
{
    template <StressFreeState stressFreeState>
    using TriangleForce = TriangleLimForce<stressFreeState>;

    real ka;
    real a3, a4;
    real mu;
    real b1, b2;
    real totArea0;     ///< equilibrium totalarea (not used for stress free case, used to compute eq length and local areas)
};
MIRHEO_MEMBER_VARS(LimParameters, ka, a3, a4, mu, b1, b2, totArea0);

/// structure containing Kanto bending parameters
struct KantorBendingParameters
{
    using DihedralForce = DihedralKantor;
    real kb, theta;
};
MIRHEO_MEMBER_VARS(KantorBendingParameters, kb, theta);

/// structure containing Juelicher bending + ADE parameters
struct JuelicherBendingParameters
{
    using DihedralForce = DihedralJuelicher;
    real kb, C0, kad, DA0;
};
MIRHEO_MEMBER_VARS(JuelicherBendingParameters, kb, C0, kad, DA0);

} // namespace mirheo
