#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/reflection.h>

#include <extern/variant/include/mpark/variant.hpp>

namespace mirheo
{

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
MIRHEO_MEMBER_VARS_4(DPDParams, a, gamma, kBT, power);

struct LJAwarenessParamsNone   {using KernelType = LJAwarenessNone;};
struct LJAwarenessParamsObject {using KernelType = LJAwarenessObject;};
struct LJAwarenessParamsRod
{
    using KernelType = LJAwarenessRod;
    int minSegmentsDist;
};
MIRHEO_MEMBER_VARS_0(LJAwarenessParamsNone);
MIRHEO_MEMBER_VARS_0(LJAwarenessParamsObject);
MIRHEO_MEMBER_VARS_1(LJAwarenessParamsRod, minSegmentsDist);

using VarLJAwarenessParams = mpark::variant<LJAwarenessParamsNone,
                                            LJAwarenessParamsObject,
                                            LJAwarenessParamsRod>;

struct LJParams
{
    real epsilon, sigma, maxForce;
    VarLJAwarenessParams varLJAwarenessParams;
};
MIRHEO_MEMBER_VARS_4(LJParams, epsilon, sigma, maxForce, varLJAwarenessParams);

struct MDPDParams
{
    using KernelType = PairwiseMDPD;
    real rd, a, b, gamma, kBT, power;
};
MIRHEO_MEMBER_VARS_6(MDPDParams, rd, a, b, gamma, kBT, power);


struct SimpleMDPDDensityKernelParams {using KernelType = SimpleMDPDDensityKernel;};
struct WendlandC2DensityKernelParams {using KernelType = WendlandC2DensityKernel;};
MIRHEO_MEMBER_VARS_0(SimpleMDPDDensityKernelParams);
MIRHEO_MEMBER_VARS_0(WendlandC2DensityKernelParams);

using VarDensityKernelParams = mpark::variant<SimpleMDPDDensityKernelParams,
                                              WendlandC2DensityKernelParams>;


struct DensityParams
{
    VarDensityKernelParams varDensityKernelParams;
};
MIRHEO_MEMBER_VARS_1(DensityParams, varDensityKernelParams);


struct LinearPressureEOSParams
{
    using KernelType = LinearPressureEOS;
    real soundSpeed, rho0;
};
MIRHEO_MEMBER_VARS_2(LinearPressureEOSParams, soundSpeed, rho0);

struct QuasiIncompressiblePressureEOSParams
{
    using KernelType = QuasiIncompressiblePressureEOS;
    real p0, rhor;
};
MIRHEO_MEMBER_VARS_2(QuasiIncompressiblePressureEOSParams, p0, rhor);

using VarEOSParams = mpark::variant<LinearPressureEOSParams,
                                    QuasiIncompressiblePressureEOSParams>;


using VarSDPDDensityKernelParams = mpark::variant<WendlandC2DensityKernelParams>;

struct SDPDParams
{
    real viscosity, kBT;
    VarEOSParams varEOSParams;
    VarSDPDDensityKernelParams varDensityKernelParams;
};
MIRHEO_MEMBER_VARS_4(SDPDParams, viscosity, kBT, varEOSParams, varDensityKernelParams);


using VarPairwiseParams = mpark::variant<DPDParams,
                                         LJParams,
                                         MDPDParams,
                                         DensityParams,
                                         SDPDParams>;


struct StressNoneParams {};
MIRHEO_MEMBER_VARS_0(StressNoneParams);

struct StressActiveParams
{
    real period; // compute stresses every this time in time units
};
MIRHEO_MEMBER_VARS_1(StressActiveParams, period);

using VarStressParams = mpark::variant<StressNoneParams, StressActiveParams>;

} // namespace mirheo
