// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

/// Describes if the shear energy kernels should fetch information from stress-free mesh
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
    real ka; ///< magnitude of total area constraint energy
    real kv; ///< magnitude of total volume constraint energy
    real gammaC;  ///< viscous coefficient, central part
    real gammaT;  ///< viscous coefficient, tengential part (will not conserve angular momentum if non zero!)
    real kBT;     ///< Temperature in energy units
    real totArea0;   ///< Total area at equilibrium
    real totVolume0; ///< Total volume at equilibrium
    bool fluctuationForces; ///< wether to enable or disable fluctuation forces
};
MIRHEO_MEMBER_VARS(CommonMembraneParameters, ka, kv, gammaC, gammaT, kBT,
                   totArea0, totVolume0, fluctuationForces);

/// structure containing WLC bond + local area energy parameters
struct WLCParameters
{
    /// The associated kernel
    template <StressFreeState stressFreeState>
    using TriangleForce = TriangleWLCForce<stressFreeState>;

    real x0; ///< normalized equilibrium length
    real ks; ///< bond energy magnitude (spring constant)
    real mpow; ///< power coefficient in wlc term
    real kd;           ///< local area energy
    real totArea0;     ///< equilibrium totalarea (not used for stress free case, used to compute eq length and local areas)
};
MIRHEO_MEMBER_VARS(WLCParameters, x0, ks, mpow, kd, totArea0);

/// structure containing Lim shear energy parameters
struct LimParameters
{
    /// The associated kernel
    template <StressFreeState stressFreeState>
    using TriangleForce = TriangleLimForce<stressFreeState>;

    real ka; ///< magnitude of expansion energy
    real a3; ///< non linear coefficient
    real a4; ///< non linear coefficient
    real mu; ///< shear energy magnitude
    real b1; ///< non linear coefficient
    real b2; ///< non linear coefficient
    real totArea0; ///< equilibrium total area (not used for stress free case, used to compute eq length and local areas)
};
MIRHEO_MEMBER_VARS(LimParameters, ka, a3, a4, mu, b1, b2, totArea0);

/// structure containing Kanto bending parameters
struct KantorBendingParameters
{
    /// The associated kernel
    using DihedralForce = DihedralKantor;
    real kb;    ///< bending energy magnitude
    real theta; ///< equilibrium dihedral angle
};
MIRHEO_MEMBER_VARS(KantorBendingParameters, kb, theta);

/// structure containing Juelicher bending + ADE parameters
struct JuelicherBendingParameters
{
    /// The associated kernel
    using DihedralForce = DihedralJuelicher;
    real kb;  ///< bending energy magnitude
    real C0;  ///< mean curvature / 2
    real kad; ///< area-difference energy magnitude
    real DA0; ///< equilibrium area difference
};
MIRHEO_MEMBER_VARS(JuelicherBendingParameters, kb, C0, kad, DA0);

} // namespace mirheo
