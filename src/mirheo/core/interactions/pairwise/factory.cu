#include "factory.h"

#include "pairwise.h"
#include "pairwise_with_stress.h"

#include "kernels/density.h"
#include "kernels/density_kernels.h"
#include "kernels/dpd.h"
#include "kernels/lj.h"
#include "kernels/mdpd.h"
#include "kernels/pressure_EOS.h"
#include "kernels/repulsive_lj.h"
#include "kernels/sdpd.h"
#include "kernels/type_traits.h"

#include <mirheo/core/utils/variant_foreach.h>

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
createPairwiseFromParams(const MirState *state, const std::string& name, real rc, const RepulsiveLJParams& params, const VarStressParams& varStressParams)
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


namespace {
struct PairwiseFactoryVisitor {
    const MirState *state;
    Loader& loader;
    const ConfigObject& config;
    const std::string& typeName;
    std::shared_ptr<BasePairwiseInteraction> impl;
};
} // anonymous namespace

/// Creates the given stress pairwise interaction if the type name matches.
template <class KernelType>
static void tryLoadPairwiseStress(PairwiseFactoryVisitor &visitor)
{
    using T = PairwiseInteractionWithStress<KernelType>;
    if (T::getTypeName() == visitor.typeName) {
        die("PairwiseInteractionWithStress load constructor not implemented. typeName = %s",
            visitor.typeName.c_str());
        // visitor.impl = std::make_shared<PairwiseInteractionWithStress<KernelType>>(
        //         visitor.state, visitor.loader, visitor.config);
    }
}

/// Creates the given no-stress pairwise interaction if the type name matches.
template <class KernelType>
static void tryLoadPairwiseNoStress(PairwiseFactoryVisitor &visitor)
{
    using T = PairwiseInteraction<KernelType>;
    if (T::getTypeName() == visitor.typeName) {
        visitor.impl = std::make_shared<PairwiseInteraction<KernelType>>(
                visitor.state, visitor.loader, visitor.config);
    }
}


std::shared_ptr<BasePairwiseInteraction>
loadInteractionPairwise(const MirState *state, Loader& loader, const ConfigObject& config)
{
    static_assert(std::is_same<
            VarPairwiseParams,
            mpark::variant<DPDParams, LJParams, RepulsiveLJParams,
                           MDPDParams, DensityParams, SDPDParams>>::value,
            "Load interactions must be updated if th VairPairwiseParams is changed.");

    const std::string& typeName = config["__type"].getString();
    PairwiseFactoryVisitor visitor{state, loader, config, typeName, nullptr};

    // The following code iterates through all possible types of interactions.
    // For interactions that branch into more template combinations, we use
    // `variantForeach`, and for those that don't we simply call
    // `tryLoadPairwise` directly.

    // DPDParams.
    tryLoadPairwiseStress  <DPDParams::KernelType>(visitor);
    tryLoadPairwiseNoStress<DPDParams::KernelType>(visitor);

    // LJParams.
    tryLoadPairwiseStress  <LJParams::KernelType>(visitor);
    tryLoadPairwiseNoStress<LJParams::KernelType>(visitor);

    // RepulsiveLJParams.
    variantForeach<VarLJAwarenessParams>([&visitor](auto type)
            {
                using T = PairwiseRepulsiveLJ<typename decltype(type)::type::KernelType>;
                tryLoadPairwiseStress  <T>(visitor);
                tryLoadPairwiseNoStress<T>(visitor);
            });

    // MDPDParams.
    tryLoadPairwiseStress  <MDPDParams::KernelType>(visitor);
    tryLoadPairwiseNoStress<MDPDParams::KernelType>(visitor);

    // DensityParams.
    variantForeach<VarDensityKernelParams>([&visitor](auto type)
            {
                using T = PairwiseDensity<typename decltype(type)::type::KernelType>;
                // tryLoadPairwiseStress<T>(visitor);  // Not applicable!
                tryLoadPairwiseNoStress<T>(visitor);
            });

    // SDPDParams.
    variantForeach<VarEOSParams, VarSDPDDensityKernelParams>([&visitor](auto eos, auto density)
            {
                using T = PairwiseSDPD<typename decltype(eos)::type::KernelType,
                                       typename decltype(density)::type::KernelType>;
                tryLoadPairwiseStress  <T>(visitor);
                tryLoadPairwiseNoStress<T>(visitor);
            });

    if (!visitor.impl)
        die("Unrecognized impl type \"%s\".", typeName.c_str());

    return std::move(visitor.impl);
}

} // namespace mirheo
