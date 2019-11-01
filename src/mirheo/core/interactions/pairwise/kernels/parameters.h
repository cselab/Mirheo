#pragma once

#include <mirheo/core/datatypes.h>

#include <extern/variant/include/mpark/variant.hpp>

// forward declaration of pairwise kernels

class PairwiseDPD;

struct LJAwarenessNone;
struct LJAwarenessObject;
struct LJAwarenessRod;

template <class Awareness>
class PairwiseRepulsiveLJ;

class PairwiseMDPD;

class SimpleMDPDDensityKernel;
class WendlandC2DensityKernel;

template <typename DensityKernel>
class PairwiseDensity;

class LinearPressureEOS;
class QuasiIncompressiblePressureEOS;

template <typename PressureEOS, typename DensityKernel>
class PairwiseSDPD;

// corresponding parameters, visible by users

struct DPDParams
{
    using KernelType = PairwiseDPD;
    real a, gamma, kBT, power;
};

struct LJAwarenessParamsNone   {using KernelType = LJAwarenessNone;};
struct LJAwarenessParamsObject {using KernelType = LJAwarenessObject;};
struct LJAwarenessParamsRod
{
    using KernelType = LJAwarenessRod;
    int minSegmentsDist;
};

using VarLJAwarenessParams = mpark::variant<LJAwarenessParamsNone,
                                            LJAwarenessParamsObject,
                                            LJAwarenessParamsRod>;

struct LJParams
{
    real epsilon, sigma, maxForce;
    VarLJAwarenessParams varLJAwarenessParams;
};

struct MDPDParams
{
    using KernelType = PairwiseMDPD;
    real rd, a, b, gamma, kBT, power;
};


struct SimpleMDPDDensityKernelParams {using KernelType = SimpleMDPDDensityKernel;};
struct WendlandC2DensityKernelParams {using KernelType = WendlandC2DensityKernel;};

using VarDensityKernelParams = mpark::variant<SimpleMDPDDensityKernelParams,
                                              WendlandC2DensityKernelParams>;


struct DensityParams
{
    VarDensityKernelParams varDensityKernelParams;
};


struct LinearPressureEOSParams
{
    using KernelType = LinearPressureEOS;
    real soundSpeed, rho0;
};

struct QuasiIncompressiblePressureEOSParams
{
    using KernelType = QuasiIncompressiblePressureEOS;
    real p0, rhor;
};

using VarEOSParams = mpark::variant<LinearPressureEOSParams,
                                    QuasiIncompressiblePressureEOSParams>;


using VarSDPDDensityKernelParams = mpark::variant<WendlandC2DensityKernelParams>;

struct SDPDParams
{
    real viscosity, kBT;
    VarEOSParams varEOSParams;
    VarSDPDDensityKernelParams varDensityKernelParams;
};


using VarPairwiseParams = mpark::variant<DPDParams,
                                         LJParams,
                                         MDPDParams,
                                         DensityParams,
                                         SDPDParams>;


struct StressNoneParams {};

struct StressActiveParams
{
    real period; // compute stresses every this time in time units
};

using VarStressParams = mpark::variant<StressNoneParams, StressActiveParams>;
