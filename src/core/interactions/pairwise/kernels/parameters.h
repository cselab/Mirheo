#pragma once

#include <extern/variant/include/mpark/variant.hpp>

// forward declaration of pairwise kernels

class PairwiseDPD;
class PairwiseLJ;
class PairwiseLJObjectAware;
class PairwiseLJRodAware;
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
    float a, gamma, kBT, power, dt;
};

struct LJAwarenessParamsNone   {};
struct LJAwarenessParamsObject {};
struct LJAwarenessParamsRod   {};

using VarLJAwarenessParams = mpark::variant<LJAwarenessParamsNone,
                                            LJAwarenessParamsObject,
                                            LJAwarenessParamsRod>;

struct LJParams
{
    float epsilon, sigma, maxForce;
    VarLJAwarenessParams varLJAwarenessParams;
};

struct MDPDParams
{
    float rd, a, b, gamma, kBT, power, dt;
};


struct SimpleMDPDDensityKernelParams {};
struct WendlandC2DensityKernelParams {};

using VarDensityKernelParams = mpark::variant<SimpleMDPDDensityKernelParams,
                                              WendlandC2DensityKernelParams>;


struct DensityParams
{
    VarDensityKernelParams densityKernelParams;
};


struct LinearPressureEOSParams
{
    float soundSpeed, rho0;
};

struct QuasiIncompressiblePressureEOSParams
{
    float p0, rhor;
};

using VarEOSParams = mpark::variant<LinearPressureEOSParams,
                                    QuasiIncompressiblePressureEOSParams>;



struct SDPDParams
{
    float viscosity, kBT, dt;
    VarEOSParams EOSParams;
    VarDensityKernelParams densityKernelParams;    
};


using VarPairwiseParams = mpark::variant<DPDParams,
                                         LJParams,
                                         MDPDParams,
                                         DensityParams,
                                         SDPDParams>;
