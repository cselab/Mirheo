#include "factory.h"

#include "membrane.h"
#include "obj_rod_binding.h"
#include "pairwise.h"
#include "pairwise/factory_helper.h"
#include "rod.h"

#include <core/logger.h>

static CommonMembraneParameters readCommonParameters(ParametersWrap& desc)
{
    CommonMembraneParameters p;

    p.totArea0    = desc.read<float>("tot_area");
    p.totVolume0  = desc.read<float>("tot_volume");

    p.ka = desc.read<float>("ka_tot");
    p.kv = desc.read<float>("kv_tot");

    p.gammaC = desc.read<float>("gammaC");
    p.gammaT = desc.read<float>("gammaT");
    p.kBT    = desc.read<float>("kBT");

    p.fluctuationForces = (p.kBT > 1e-6);
    
    return p;
}

static WLCParameters readWLCParameters(ParametersWrap& desc)
{
    WLCParameters p;

    p.x0   = desc.read<float>("x0");
    p.ks   = desc.read<float>("ks");
    p.mpow = desc.read<float>("mpow");

    p.kd = desc.read<float>("ka");
    p.totArea0 = desc.read<float>("tot_area");
    
    return p;
}

static LimParameters readLimParameters(ParametersWrap& desc)
{
    LimParameters p;

    p.ka = desc.read<float>("ka");
    p.a3 = desc.read<float>("a3");
    p.a4 = desc.read<float>("a4");
    
    p.mu = desc.read<float>("mu");
    p.b1 = desc.read<float>("b1");
    p.b2 = desc.read<float>("b2");

    p.totArea0 = desc.read<float>("tot_area");
    
    return p;
}

static KantorBendingParameters readKantorParameters(ParametersWrap& desc)
{
    KantorBendingParameters p;

    p.kb    = desc.read<float>("kb");
    p.theta = desc.read<float>("theta");
    
    return p;
}

static JuelicherBendingParameters readJuelicherParameters(ParametersWrap& desc)
{
    JuelicherBendingParameters p;

    p.kb = desc.read<float>("kb");
    p.C0 = desc.read<float>("C0");

    p.kad = desc.read<float>("kad");
    p.DA0 = desc.read<float>("DA0");
    
    return p;
}

std::shared_ptr<MembraneInteraction>
InteractionFactory::createInteractionMembrane(const MirState *state, std::string name,
                                              std::string shearDesc, std::string bendingDesc,
                                              const MapParams& parameters,
                                              bool stressFree, float growUntil)
{
    VarBendingParams bendingParams;
    VarShearParams shearParams;
    ParametersWrap desc {parameters};    
    
    auto commonPrms = readCommonParameters(desc);

    if      (shearDesc == "wlc") shearParams = readWLCParameters(desc);
    else if (shearDesc == "Lim") shearParams = readLimParameters(desc);
    else                         die("No such shear parameters: '%s'", shearDesc.c_str());

    if      (bendingDesc == "Kantor")    bendingParams = readKantorParameters(desc);
    else if (bendingDesc == "Juelicher") bendingParams = readJuelicherParameters(desc);
    else                                 die("No such bending parameters: '%s'", bendingDesc.c_str());

    desc.checkAllRead();
    return std::make_shared<MembraneInteraction>
        (state, name, commonPrms, bendingParams, shearParams, stressFree, growUntil);
}

static RodParameters readRodParameters(ParametersWrap& desc)
{
    RodParameters p;

    if (desc.exists<std::vector<PyTypes::float2>>( "kappa0" ))
    {
        auto kappaEqs = desc.read<std::vector<PyTypes::float2>>( "kappa0");
        auto tauEqs   = desc.read<std::vector<float>>( "tau0");
        auto groundE  = desc.read<std::vector<float>>( "E0");

        if (kappaEqs.size() != tauEqs.size() || tauEqs.size() != groundE.size())
            die("Rod parameters: expected same number of kappa0, tau0 and E0");

        for (const auto& om : kappaEqs)
            p.kappaEq.push_back(make_float2(om));
        
        for (const auto& tau : tauEqs)
            p.tauEq.push_back(tau);

        for (const auto& E : groundE)
            p.groundE.push_back(E);
    }
    else
    {
        p.kappaEq.push_back(desc.read<float2>("kappa0"));
        p.tauEq  .push_back(desc.read<float>("tau0"));

        if (desc.exists<float>("E0"))
            p.groundE.push_back(desc.read<float>("E0"));
        else
            p.groundE.push_back(0.f);
    }
    
    p.kBending  = desc.read<float3>("k_bending");
    p.kTwist    = desc.read<float>("k_twist");
    
    p.a0        = desc.read<float>("a0");
    p.l0        = desc.read<float>("l0");
    p.ksCenter  = desc.read<float>("k_s_center");
    p.ksFrame   = desc.read<float>("k_s_frame");
    return p;
}

static StatesSmoothingParameters readStatesSmoothingRodParameters(ParametersWrap& desc)
{
    StatesSmoothingParameters p;
    p.kSmoothing = desc.read<float>("k_smoothing");
    return p;
}

static StatesSpinParameters readStatesSpinRodParameters(ParametersWrap& desc)
{
    StatesSpinParameters p;

    p.nsteps = desc.read<float>("nsteps");
    p.kBT    = desc.read<float>("kBT");
    p.J      = desc.read<float>("J");
    return p;
}


std::shared_ptr<RodInteraction>
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
    return std::make_shared<RodInteraction>(state, name, params, spinParams, saveEnergies);
}

std::shared_ptr<PairwiseInteraction>
InteractionFactory::createPairwiseInteraction(const MirState *state, std::string name, float rc, const std::string type, const MapParams& parameters)
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
    return std::make_shared<PairwiseInteraction>(state, name, rc, varParams, varStressParams);
}

std::shared_ptr<ObjectRodBindingInteraction>
InteractionFactory::createInteractionObjRodBinding(const MirState *state, std::string name,
                                                   float torque, PyTypes::float3 relAnchor, float kBound)
{
    return std::make_shared<ObjectRodBindingInteraction>(state, name, torque, make_float3(relAnchor), kBound);
}
