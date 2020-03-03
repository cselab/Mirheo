#include "factory.h"

#include "membrane/base_membrane.h"
#include "membrane/factory.h"
#include "obj_rod_binding.h"
#include "pairwise/base_pairwise.h"
#include "pairwise/factory.h"
#include "pairwise/factory_helper.h"
#include "rod/base_rod.h"
#include "rod/factory.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/config.h>

namespace mirheo
{

static CommonMembraneParameters readCommonParameters(ParametersWrap& desc)
{
    CommonMembraneParameters p;

    p.totArea0    = desc.read<real>("tot_area");
    p.totVolume0  = desc.read<real>("tot_volume");

    p.ka = desc.read<real>("ka_tot");
    p.kv = desc.read<real>("kv_tot");

    p.gammaC = desc.read<real>("gammaC");
    p.gammaT = desc.read<real>("gammaT");
    p.kBT    = desc.read<real>("kBT");

    p.fluctuationForces = (p.kBT > 1e-6_r);
    
    return p;
}

static WLCParameters readWLCParameters(ParametersWrap& desc)
{
    WLCParameters p;

    p.x0   = desc.read<real>("x0");
    p.ks   = desc.read<real>("ks");
    p.mpow = desc.read<real>("mpow");

    p.kd = desc.read<real>("ka");
    p.totArea0 = desc.read<real>("tot_area");
    
    return p;
}

static LimParameters readLimParameters(ParametersWrap& desc)
{
    LimParameters p;

    p.ka = desc.read<real>("ka");
    p.a3 = desc.read<real>("a3");
    p.a4 = desc.read<real>("a4");
    
    p.mu = desc.read<real>("mu");
    p.b1 = desc.read<real>("b1");
    p.b2 = desc.read<real>("b2");

    p.totArea0 = desc.read<real>("tot_area");
    
    return p;
}

static KantorBendingParameters readKantorParameters(ParametersWrap& desc)
{
    KantorBendingParameters p;

    p.kb    = desc.read<real>("kb");
    p.theta = desc.read<real>("theta");
    
    return p;
}

static JuelicherBendingParameters readJuelicherParameters(ParametersWrap& desc)
{
    JuelicherBendingParameters p;

    p.kb = desc.read<real>("kb");
    p.C0 = desc.read<real>("C0");

    p.kad = desc.read<real>("kad");
    p.DA0 = desc.read<real>("DA0");
    
    return p;
}

static FilterKeepByTypeId readFilterKeepByTypeId(ParametersWrap& desc)
{
    const int typeId = static_cast<int>(desc.read<real>("type_id"));
    return FilterKeepByTypeId{typeId};
}

std::shared_ptr<BaseMembraneInteraction>
InteractionFactory::createInteractionMembrane(const MirState *state, std::string name,
                                              std::string shearDesc, std::string bendingDesc,
                                              std::string filterDesc, const MapParams& parameters,
                                              bool stressFree)
{
    VarBendingParams varBendingParams;
    VarShearParams varShearParams;
    VarMembraneFilter varFilter;
    ParametersWrap desc {parameters};    

    // those are default parameters
    real initLengthFraction {1.0_r};
    real growUntil          {0.0_r};
    
    auto commonPrms = readCommonParameters(desc);

    if      (shearDesc == "wlc") varShearParams = readWLCParameters(desc);
    else if (shearDesc == "Lim") varShearParams = readLimParameters(desc);
    else                         die("No such shear parameters: '%s'", shearDesc.c_str());

    if      (bendingDesc == "Kantor")    varBendingParams = readKantorParameters(desc);
    else if (bendingDesc == "Juelicher") varBendingParams = readJuelicherParameters(desc);
    else                                 die("No such bending parameters: '%s'", bendingDesc.c_str());

    if      (filterDesc == "keep_all")   varFilter = FilterKeepAll{};
    else if (filterDesc == "by_type_id") varFilter = readFilterKeepByTypeId(desc);
    else                                 die("No such filter parameters: '%s'", filterDesc.c_str());

    if (desc.exists<real>("grow_until") || desc.exists<real>("init_length_fraction"))
    {
        growUntil          = desc.read<real>("grow_until");
        initLengthFraction = desc.read<real>("init_length_fraction");
    }
    
    desc.checkAllRead();
    return createInteractionMembrane(
        state, name, commonPrms, varBendingParams, varShearParams, stressFree,
        initLengthFraction, growUntil, varFilter);
}

static RodParameters readRodParameters(ParametersWrap& desc)
{
    RodParameters p;

    if (desc.exists<std::vector<real2>>( "kappa0" ))
    {
        auto kappaEqs = desc.read<std::vector<real2>>( "kappa0");
        auto tauEqs   = desc.read<std::vector<real>>( "tau0");
        auto groundE  = desc.read<std::vector<real>>( "E0");

        if (kappaEqs.size() != tauEqs.size() || tauEqs.size() != groundE.size())
            die("Rod parameters: expected same number of kappa0, tau0 and E0");

        for (const auto& om : kappaEqs)
            p.kappaEq.push_back(om);
        
        for (const auto& tau : tauEqs)
            p.tauEq.push_back(tau);

        for (const auto& E : groundE)
            p.groundE.push_back(E);
    }
    else
    {
        p.kappaEq.push_back(desc.read<real2>("kappa0"));
        p.tauEq  .push_back(desc.read<real>("tau0"));

        if (desc.exists<real>("E0"))
            p.groundE.push_back(desc.read<real>("E0"));
        else
            p.groundE.push_back(0._r);
    }
    
    p.kBending  = desc.read<real3>("k_bending");
    p.kTwist    = desc.read<real>("k_twist");
    
    p.a0        = desc.read<real>("a0");
    p.l0        = desc.read<real>("l0");
    p.ksCenter  = desc.read<real>("k_s_center");
    p.ksFrame   = desc.read<real>("k_s_frame");
    return p;
}

static StatesSmoothingParameters readStatesSmoothingRodParameters(ParametersWrap& desc)
{
    StatesSmoothingParameters p;
    p.kSmoothing = desc.read<real>("k_smoothing");
    return p;
}

static StatesSpinParameters readStatesSpinRodParameters(ParametersWrap& desc)
{
    StatesSpinParameters p;

    p.nsteps = static_cast<int>(desc.read<real>("nsteps"));
    p.kBT    = desc.read<real>("kBT");
    p.J      = desc.read<real>("J");
    return p;
}


std::shared_ptr<BaseRodInteraction>
InteractionFactory::createInteractionRod(const MirState *state, std::string name, std::string stateUpdate,
                                         bool saveEnergies, const MapParams& parameters)
{
    ParametersWrap desc {parameters};
    auto params = readRodParameters(desc);

    VarSpinParams spinParams;
    
    if      (stateUpdate == "none")
        spinParams = StatesParametersNone{};
    else if (stateUpdate == "smoothing")
        spinParams = readStatesSmoothingRodParameters(desc);
    else if (stateUpdate == "spin")
        spinParams = readStatesSpinRodParameters(desc);
    else
        die("unrecognised state update method: '%s'", stateUpdate.c_str());
    
    desc.checkAllRead();
    return createInteractionRod(state, name, params, spinParams, saveEnergies);
}

std::shared_ptr<BasePairwiseInteraction>
InteractionFactory::createPairwiseInteraction(const MirState *state, std::string name, real rc, const std::string type, const MapParams& parameters)
{
    ParametersWrap desc {parameters};
    VarPairwiseParams varParams;
    
    if (type == "DPD")
        varParams = FactoryHelper::readDPDParams(desc);
    else if (type == "MDPD")
        varParams = FactoryHelper::readMDPDParams(desc);
    else if (type == "SDPD")
        varParams = FactoryHelper::readSDPDParams(desc);
    else if (type == "RepulsiveLJ")
        varParams = FactoryHelper::readLJParams(desc);
    else if (type == "Density")
        varParams = FactoryHelper::readDensityParams(desc);
    else
        die("Unrecognized pairwise interaction type '%s'", type.c_str());

    const auto varStressParams = FactoryHelper::readStressParams(desc);

    desc.checkAllRead();
    return createInteractionPairwise(state, name, rc, varParams, varStressParams);
}

std::shared_ptr<ObjectRodBindingInteraction>
InteractionFactory::createInteractionObjRodBinding(const MirState *state, std::string name,
                                                   real torque, real3 relAnchor, real kBound)
{
    return std::make_shared<ObjectRodBindingInteraction>(state, name, torque, relAnchor, kBound);
}

static bool startsWith(const std::string &text, const char *tmp)
{
    return text.compare(0, strlen(tmp), tmp) == 0;
}

std::shared_ptr<Interaction>
InteractionFactory::loadInteraction(const MirState *state, Loader& loader, const ConfigObject& config)
{
    const std::string& type = config["__type"];
    if (startsWith(type, "PairwiseInteraction<"))
        return loadInteractionPairwise(state, loader, config);
    if (startsWith(type, "MembraneInteraction<"))
        return loadInteractionMembrane(state, loader, config);
    die("Unrecognized or unimplemented interaction type \"%s\".", type.c_str());
}

} // namespace mirheo
