// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/variant.h>

namespace mirheo
{

// forward declaration of pairwise kernels

class PairwiseDPD;
class PairwiseNoRandomDPD;
class PairwiseLJ;

struct AwarenessNone;
struct AwarenessObject;
struct AwarenessRod;

template <class Awareness>
class PairwiseRepulsiveLJ;

template <class Awareness>
class PairwiseGrowingRepulsiveLJ;

class PairwiseMDPD;

template <class Awareness>
class PairwiseMorse;

class SimpleMDPDDensityKernel;
class WendlandC2DensityKernel;

template <typename DensityKernel>
class PairwiseDensity;

class LinearPressureEOS;
class QuasiIncompressiblePressureEOS;

template <typename PressureEOS, typename DensityKernel>
class PairwiseSDPD;

// corresponding parameters, visible by users

/// Dissipative Particle Dynamics  parameters
struct DPDParams
{
    using KernelType = PairwiseDPD; ///< the corresponding kernel
    real a;     ///< conservative force coefficient
    real gamma; ///< dissipative force conservative
    real kBT;   ///< temperature in energy units
    real power; ///< exponent of the envelope of the viscous kernel
};

/// Dissipative Particle Dynamics parameters with no fluctuations
struct NoRandomDPDParams
{
    using KernelType = PairwiseNoRandomDPD; ///< the corresponding kernel
    real a;     ///< conservative force coefficient
    real gamma; ///< dissipative force conservative
    real kBT;   ///< temperature in energy units
    real power; ///< exponent of the envelope of the viscous kernel
};

/// Lennard-Jones parameters
struct LJParams
{
    using KernelType = PairwiseLJ; ///< the corresponding kernel
    real epsilon; ///< force coefficient
    real sigma;   ///< radius with zero energy in LJ potential
};

/// Parameters for no awareness in pairwise interactions
struct AwarenessParamsNone
{
    using KernelType = AwarenessNone; ///< the corresponding kernel
};

/// Parameters for object awareness in pairwise interactions
struct AwarenessParamsObject
{
    using KernelType = AwarenessObject; ///< the corresponding kernel
};

/// Parameters for rod awareness in pairwise interactions
struct AwarenessParamsRod
{
    using KernelType = AwarenessRod; ///< the corresponding kernel
    int minSegmentsDist; ///< number of segments away to ignore the self interaction
};

/// variant of all awareness modes
using VarAwarenessParams = mpark::variant<AwarenessParamsNone,
                                          AwarenessParamsObject,
                                          AwarenessParamsRod>;


/// Repulsive Lennard-Jones parameters
struct RepulsiveLJParams
{
    real epsilon;  ///< force coefficient
    real sigma;    ///< radius with zero energy in LJ potential
    real maxForce; ///< cap force
    VarAwarenessParams varAwarenessParams; ///< awareness
};

/// Growing Repulsive Lennard-Jones parameters
struct GrowingRepulsiveLJParams
{
    real epsilon;  ///< force coefficient
    real sigma;    ///< radius with zero energy in LJ potential
    real maxForce; ///< cap force
    VarAwarenessParams varAwarenessParams; ///< awareness
    real initialLengthFraction; ///< initial factor for the length scale
    real growUntil; ///< time after which the length factor is one
};


/// Morse parameters
struct MorseParams
{
    real De; ///< force coefficient
    real r0; ///< zero force distance
    real beta; ///< interaction range parameter
    VarAwarenessParams varAwarenessParams; ///< awareness
};


/// Multi-body Dissipative Particle Dynamics parameters
struct MDPDParams
{
    using KernelType = PairwiseMDPD; ///< the corresponding kernel
    real rd;    ///< density cut-off radius
    real a;     ///< conservative force coefficient (repulsive)
    real b;     ///< conservative force coefficient (attractive)
    real gamma; ///< dissipative force conservative
    real kBT;   ///< temperature in energy units
    real power; ///< exponent of the envelope of the viscous kernel
};

/// Density parameters for MDPD
struct SimpleMDPDDensityKernelParams
{
    using KernelType = SimpleMDPDDensityKernel; ///< the corresponding kernel
};

/// Density parameters for Wendland C2 function
struct WendlandC2DensityKernelParams
{
    using KernelType = WendlandC2DensityKernel; ///< the corresponding kernel
};

/// variant of all density types
using VarDensityKernelParams = mpark::variant<SimpleMDPDDensityKernelParams,
                                              WendlandC2DensityKernelParams>;

/// Density parameters
struct DensityParams
{
    VarDensityKernelParams varDensityKernelParams; ///< kernel parameters
};


/// parameters for linear equation of state
struct LinearPressureEOSParams
{
    using KernelType = LinearPressureEOS; ///< the corresponding kernel
    real soundSpeed; ///< Speed of sound
    real rho0;       ///< reference density
};

/// parameters for quasi incompressible equation of state
struct QuasiIncompressiblePressureEOSParams
{
    using KernelType = QuasiIncompressiblePressureEOS;  ///< the corresponding kernel
    real p0;   ///< pressure magnitude
    real rhor; ///< reference density
};

/// variant of all equation of states parameters
using VarEOSParams = mpark::variant<LinearPressureEOSParams,
                                    QuasiIncompressiblePressureEOSParams>;

/// variant of all density kernels compatible with SDPD
using VarSDPDDensityKernelParams = mpark::variant<WendlandC2DensityKernelParams>;

/// Smoothed Dissipative Particle Dynamics parameters
struct SDPDParams
{
    real viscosity; ///< dynamic viscosity of the fluid
    real kBT;       ///< temperature in energy units
    VarEOSParams varEOSParams; ///< equation of state
    VarSDPDDensityKernelParams varDensityKernelParams; ///< density kernel
};

/// variant of all possible pairwise interactions
using VarPairwiseParams = mpark::variant<DPDParams,
                                         LJParams,
                                         MorseParams,
                                         RepulsiveLJParams,
                                         GrowingRepulsiveLJParams,
                                         MDPDParams,
                                         DensityParams,
                                         SDPDParams>;


/// parameters when the stress is not active
struct StressNoneParams {};

/// parameters when the stress is active
struct StressActiveParams
{
    real period; ///< compute stresses every this time in time units
};

/// active/non active stress parameters
using VarStressParams = mpark::variant<StressNoneParams, StressActiveParams>;

} // namespace mirheo
