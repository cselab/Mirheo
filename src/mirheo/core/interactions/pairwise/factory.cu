#include "factory.h"

#include "pairwise.h"
#include "pairwise_with_stress.h"

#include "kernels/density.h"
#include "kernels/density_kernels.h"
#include "kernels/dpd.h"
#include "kernels/mdpd.h"
#include "kernels/pressure_EOS.h"
#include "kernels/repulsive_lj.h"
#include "kernels/sdpd.h"
#include "kernels/type_traits.h"

namespace mirheo
{

template <class KernelType>
static std::shared_ptr<BasePairwiseInteraction>
createPairwiseFromKernel(const MirState *state, const std::string& name, real rc,
                         const typename KernelType::ParamsType& params, const VarStressParams& varStressParams)
{
    if (mpark::holds_alternative<StressActiveParams>(varStressParams))
    {
        const auto stressParams = mpark::get<StressActiveParams>(varStressParams);
        return std::make_shared<PairwiseInteractionWithStress<KernelType>>(state, name, rc, stressParams.period, params);
    }
    else
    {
        return std::make_shared<PairwiseInteraction<KernelType>>(state, name, rc, params);
    }
}

template <class KernelType>
static std::shared_ptr<BasePairwiseInteraction>
createPairwiseFromKernelNoStress(const MirState *state, const std::string& name, real rc,
                                 const typename KernelType::ParamsType& params, const VarStressParams& varStressParams)
{
    if (mpark::holds_alternative<StressActiveParams>(varStressParams))
        die("Incompatible interaction output: '%s' can not output stresses.", name.c_str());
    
    return std::make_shared<PairwiseInteraction<KernelType>>(state, name, rc, params);
}


template <class Parameters>
static std::shared_ptr<BasePairwiseInteraction>
createPairwiseFromParams(const MirState *state, const std::string& name, real rc, const Parameters& params, const VarStressParams& varStressParams)
{
    using KernelType = typename Parameters::KernelType;
    return createPairwiseFromKernel<KernelType>(state, name, rc, params, varStressParams);
}


static std::shared_ptr<BasePairwiseInteraction>
createPairwiseFromParams(const MirState *state, const std::string& name, real rc, const LJParams& params, const VarStressParams& varStressParams)
{
    return mpark::visit([&](auto& awareParams)
    {
        using AwareType = typename std::remove_reference<decltype(awareParams)>::type::KernelType;
        using KernelType = PairwiseRepulsiveLJ<AwareType>;

        return createPairwiseFromKernel<KernelType>(state, name, rc, params, varStressParams);
    }, params.varLJAwarenessParams);
}

static std::shared_ptr<BasePairwiseInteraction>
createPairwiseFromParams(const MirState *state, const std::string& name, real rc, const DensityParams& params, const VarStressParams& varStressParams)
{
    return mpark::visit([&](auto& densityKernelParams)
    {
        using DensityKernelType = typename std::remove_reference<decltype(densityKernelParams)>::type::KernelType;
        using KernelType = PairwiseDensity<DensityKernelType>;;

        return createPairwiseFromKernelNoStress<KernelType>(state, name, rc, params, varStressParams);
    }, params.varDensityKernelParams);
}

static std::shared_ptr<BasePairwiseInteraction>
createPairwiseFromParams(const MirState *state, const std::string& name, real rc, const SDPDParams& params, const VarStressParams& varStressParams)
{
    return mpark::visit([&](auto& densityKernelParams, auto& EOSParams)
    {
        using DensityKernelType = typename std::remove_reference<decltype(densityKernelParams)>::type::KernelType;
        using EOSKernelType     = typename std::remove_reference<decltype(EOSParams          )>::type::KernelType;
        using KernelType = PairwiseSDPD<EOSKernelType, DensityKernelType>;
        
        return createPairwiseFromKernel<KernelType>(state, name, rc, params, varStressParams);
    }, params.varDensityKernelParams, params.varEOSParams);
}


std::shared_ptr<BasePairwiseInteraction>
createInteractionPairwise(const MirState *state, const std::string& name, real rc,
                          const VarPairwiseParams& varParams, const VarStressParams& varStressParams)
{
    return mpark::visit([&](const auto& params)
    {
        return createPairwiseFromParams(state, name, rc, params, varStressParams);
    }, varParams);
}


} // namespace mirheo
