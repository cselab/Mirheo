#include "pairwise.h"
#include "pairwise/impl.h"
#include "pairwise/impl.stress.h"

#include "pairwise/kernels/dpd.h"
#include "pairwise/kernels/density.h"
#include "pairwise/kernels/density_kernels.h"
#include "pairwise/kernels/lj.h"
#include "pairwise/kernels/mdpd.h"
#include "pairwise/kernels/pressure_EOS.h"
#include "pairwise/kernels/sdpd.h"

#include "pairwise/kernels/type_traits.h"


#include <memory>

template <class KernelType>
static std::unique_ptr<Interaction>
createPairwiseFromKernel(const MirState *state, const std::string& name, float rc,
                         const typename std::enable_if<outputsForce<KernelType>::value, KernelType>::type& kernel,
                         bool stress, float stressPeriod)
{
    if (stress)
        return std::make_unique<InteractionPair_withStress<KernelType>>(state, name, rc, stressPeriod, kernel);
    else
        return std::make_unique<InteractionPair<KernelType>>(state, name, rc, kernel);
}

template <class KernelType>
static std::unique_ptr<Interaction>
createPairwiseFromKernel(const MirState *state, const std::string& name, float rc, const KernelType& kernel,
                         __UNUSED bool stress, __UNUSED float stressPeriod)
{
    return std::make_unique<InteractionPair<KernelType>>(state, name, rc, kernel);
}


template <class Parameters>
static std::unique_ptr<Interaction>
createPairwiseFromParams(const MirState *state, const std::string& name, float rc, const Parameters& params, bool stress, float stressPeriod)
{
    using KernelType = typename Parameters::KernelType;
    KernelType kernel(rc, params);

    return createPairwiseFromKernel(state, name, rc, kernel, stress, stressPeriod);
}


std::unique_ptr<Interaction>
createPairwiseFromParams(const MirState *state, const std::string& name, float rc, const LJParams& params, bool stress, float stressPeriod)
{
    return mpark::visit([&](auto& awareParams)
    {
        using AwareType = typename std::remove_reference<decltype(awareParams)>::type::KernelType;
        
        AwareType awareness(awareParams);
        PairwiseLJ<AwareType> lj(rc, params.epsilon, params.sigma, params.maxForce, awareness);

        return createPairwiseFromKernel(state, name, rc, lj, stress, stressPeriod);
    }, params.varLJAwarenessParams);
}

static std::unique_ptr<Interaction>
createPairwiseFromParams(const MirState *state, const std::string& name, float rc, const DensityParams& params, bool stress, float stressPeriod)
{
    return mpark::visit([&](auto& densityKernelParams)
    {
        using DensityKernelType = typename std::remove_reference<decltype(densityKernelParams)>::type::KernelType;
        
        DensityKernelType densityKernel;
        PairwiseDensity<DensityKernelType> density(rc, densityKernel);

        return createPairwiseFromKernel(state, name, rc, density, stress, stressPeriod);
    }, params.varDensityKernelParams);
}

static std::unique_ptr<Interaction>
createPairwiseFromParams(const MirState *state, const std::string& name, float rc, const SDPDParams& params, bool stress, float stressPeriod)
{
    return mpark::visit([&](auto& densityKernelParams, auto& EOSParams)
    {
        using DensityKernelType = typename std::remove_reference<decltype(densityKernelParams)>::type::KernelType;
        using EOSKernelType     = typename std::remove_reference<decltype(EOSParams          )>::type::KernelType;
        
        DensityKernelType density;
        EOSKernelType pressure(EOSParams);

        PairwiseSDPD<EOSKernelType, DensityKernelType> sdpd(rc, pressure, density, params.viscosity, params.kBT, params.dt);
        
        return createPairwiseFromKernel(state, name, rc, sdpd, stress, stressPeriod);
    }, params.varDensityKernelParams, params.varEOSParams);
}


PairwiseInteraction::PairwiseInteraction(const MirState *state, const std::string& name, float rc,
                                         VarPairwiseParams varParams, bool stress, float stressPeriod) :
    Interaction(state, name, rc),
    varParams(varParams)
{
    impl = mpark::visit([&](const auto& params)
    {
        return createPairwiseFromParams(state, name, rc, params, stress, stressPeriod);
    }, varParams);
}

PairwiseInteraction::~PairwiseInteraction() = default;


