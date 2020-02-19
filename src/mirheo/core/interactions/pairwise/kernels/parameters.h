#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/reflection.h>

#include <extern/variant/include/mpark/variant.hpp>

namespace mirheo
{

// forward declaration of pairwise kernels

class PairwiseDPD;
class PairwiseNoRandomDPD;

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
MIRHEO_MEMBER_VARS(DPDParams, a, gamma, kBT, power);

struct NoRandomDPDParams
{
    using KernelType = PairwiseNoRandomDPD;
    real a, gamma, kBT, power;
};
MIRHEO_MEMBER_VARS(NoRandomDPDParams, a, gamma, kBT, power);

struct LJAwarenessParamsNone   {using KernelType = LJAwarenessNone;};
struct LJAwarenessParamsObject {using KernelType = LJAwarenessObject;};
struct LJAwarenessParamsRod
{
    using KernelType = LJAwarenessRod;
    int minSegmentsDist;
};
MIRHEO_MEMBER_VARS(LJAwarenessParamsNone);
MIRHEO_MEMBER_VARS(LJAwarenessParamsObject);
MIRHEO_MEMBER_VARS(LJAwarenessParamsRod, minSegmentsDist);

using VarLJAwarenessParams = mpark::variant<LJAwarenessParamsNone,
                                            LJAwarenessParamsObject,
                                            LJAwarenessParamsRod>;

struct LJParams
{
    real epsilon, sigma, maxForce;
    VarLJAwarenessParams varLJAwarenessParams;
};
MIRHEO_MEMBER_VARS(LJParams, epsilon, sigma, maxForce, varLJAwarenessParams);

struct MDPDParams
{
    using KernelType = PairwiseMDPD;
    real rd, a, b, gamma, kBT, power;
};
MIRHEO_MEMBER_VARS(MDPDParams, rd, a, b, gamma, kBT, power);


struct SimpleMDPDDensityKernelParams {using KernelType = SimpleMDPDDensityKernel;};
struct WendlandC2DensityKernelParams {using KernelType = WendlandC2DensityKernel;};
MIRHEO_MEMBER_VARS(SimpleMDPDDensityKernelParams);
MIRHEO_MEMBER_VARS(WendlandC2DensityKernelParams);

using VarDensityKernelParams = mpark::variant<SimpleMDPDDensityKernelParams,
                                              WendlandC2DensityKernelParams>;


struct DensityParams
{
    VarDensityKernelParams varDensityKernelParams;
};
MIRHEO_MEMBER_VARS(DensityParams, varDensityKernelParams);


struct LinearPressureEOSParams
{
    using KernelType = LinearPressureEOS;
    real soundSpeed, rho0;
};
MIRHEO_MEMBER_VARS(LinearPressureEOSParams, soundSpeed, rho0);

struct QuasiIncompressiblePressureEOSParams
{
    using KernelType = QuasiIncompressiblePressureEOS;
    real p0, rhor;
};
MIRHEO_MEMBER_VARS(QuasiIncompressiblePressureEOSParams, p0, rhor);

using VarEOSParams = mpark::variant<LinearPressureEOSParams,
                                    QuasiIncompressiblePressureEOSParams>;


using VarSDPDDensityKernelParams = mpark::variant<WendlandC2DensityKernelParams>;

struct SDPDParams
{
    real viscosity, kBT;
    VarEOSParams varEOSParams;
    VarSDPDDensityKernelParams varDensityKernelParams;
};
MIRHEO_MEMBER_VARS(SDPDParams, viscosity, kBT, varEOSParams, varDensityKernelParams);


using VarPairwiseParams = mpark::variant<DPDParams,
                                         LJParams,
                                         MDPDParams,
                                         DensityParams,
                                         SDPDParams>;


struct StressNoneParams {};
MIRHEO_MEMBER_VARS(StressNoneParams);

struct StressActiveParams
{
    real period; // compute stresses every this time in time units
};
MIRHEO_MEMBER_VARS(StressActiveParams, period);

using VarStressParams = mpark::variant<StressNoneParams, StressActiveParams>;

} // namespace mirheo
