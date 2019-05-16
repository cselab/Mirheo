#include "factory.h"

#include "density.h"
#include "dpd.h"
#include "dpd_with_stress.h"
#include "lj.h"
#include "lj_with_stress.h"
#include "mdpd.h"
#include "mdpd_with_stress.h"
#include "membrane.h"
#include "obj_rod_binding.h"
#include "pairwise_interactions/density_kernels.h"
#include "pairwise_interactions/pressure_EOS.h"
#include "rod.h"
#include "sdpd.h"
#include "sdpd_with_stress.h"

#include <core/logger.h>

using InteractionFactory::VarParam;
using MapParams = std::map<std::string, VarParam>;

class ParametersWrap
{
public:
    ParametersWrap(const MapParams& params) :
        params(params)
    {
        for (const auto& p : params)
            readParams[p.first] = false;
    }

    ~ParametersWrap()
    {
        check();
    }

    template <typename T>
    bool exists(const std::string& key)
    {
        auto it = params.find(key);

        if (it == params.end())
            return false;

        if (!mpark::holds_alternative<T>(it->second))
            return false;

        return true;
    }
    
    template <typename T>
    T read(const std::string& key)
    {
        auto it = params.find(key);
    
        if (it == params.end())
            die("missing parameter '%s'", key.c_str());

        if (!mpark::holds_alternative<T>(it->second))
            die("'%s': invalid type", key.c_str());

        readParams[key] = true;
        return mpark::get<T>(it->second);
    }

    void check() const
    {
        for (const auto& p : readParams)
            if (p.second == false)
                die("invalid parameter '%s'", p.first.c_str());
    }
    
private:
    const MapParams& params;
    std::map<std::string, bool> readParams;
};

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

std::shared_ptr<InteractionMembrane>
InteractionFactory::createInteractionMembrane(const YmrState *state, std::string name,
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

    return std::make_shared<InteractionMembrane>
        (state, name, commonPrms, bendingParams, shearParams, stressFree, growUntil);
}

static RodParameters readRodParameters(ParametersWrap& desc)
{
    RodParameters p;

    if (desc.exists<std::vector<float>>( "tau0" ))
    {
        auto omegaEqs = desc.read<std::vector<PyTypes::float2>>( "omega0");
        auto tauEqs   = desc.read<std::vector<float>>( "tau0");
        auto groundE  = desc.read<std::vector<float>>( "E0");

        if (omegaEqs.size() != tauEqs.size())
            die("Rod parameters: expected same number of omega0 and tau0");

        for (const auto& om : omegaEqs)
            p.omegaEq.push_back(make_float2(om));
        
        for (const auto& tau : tauEqs)
            p.tauEq.push_back(tau);

        for (const auto& E : groundE)
            p.groundE.push_back(E);
    }
    else
    {
        p.omegaEq.push_back(make_float2(desc.read<PyTypes::float2>("omega0")));
        p.tauEq  .push_back(desc.read<float>("tau0"));

        if (desc.exists<float>("E0"))
            p.groundE.push_back(desc.read<float>("E0"));
        else
            p.groundE.push_back(0.f);
    }
    
    p.kBending  = make_float3(desc.read<PyTypes::float3>("k_bending"));
    p.kTwist    = desc.read<float>("k_twist");
    
    p.a0        = desc.read<float>("a0");
    p.l0        = desc.read<float>("l0");
    p.kBounds   = desc.read<float>("k_bounds");
    p.kVisc     = desc.read<float>("k_visc");
    return p;
}

std::shared_ptr<InteractionRod>
InteractionFactory::createInteractionRod(const YmrState *state, std::string name,
                                         const MapParams& parameters)
{
    ParametersWrap desc {parameters};
    auto params = readRodParameters(desc);
    return std::make_shared<InteractionRod>(state, name, params);
}


static bool isSimpleMDPDDensity(const std::string& desc)
{
    return desc == "MDPD";
}


static bool isWendlandC2Density(const std::string& desc)
{
    return desc == "WendlandC2";
}

std::shared_ptr<BasicInteractionDensity>
InteractionFactory::createPairwiseDensity(const YmrState *state, std::string name, float rc,
                                          const std::string& density)
{
    if (isSimpleMDPDDensity(density))
    {
        SimpleMDPDDensityKernel densityKernel;
        return std::make_shared<InteractionDensity<SimpleMDPDDensityKernel>>
                                (state, name, rc, densityKernel);
    }
    
    if (isWendlandC2Density(density))
    {
        WendlandC2DensityKernel densityKernel;
        return std::make_shared<InteractionDensity<WendlandC2DensityKernel>>
                                (state, name, rc, densityKernel);
    }

    die("Invalid density '%s'", density.c_str());
    return nullptr;
}


static LinearPressureEOS readLinearPressureEOS(ParametersWrap& desc)
{
    float c = desc.read<float>("sound_speed");
    float r = desc.read<float>("rho_0");
    return LinearPressureEOS(c, r);
}

static QuasiIncompressiblePressureEOS readQuasiIncompressiblePressureEOS(ParametersWrap& desc)
{
    float p0   = desc.read<float>("p0");
    float rhor = desc.read<float>("rho_r");
    
    return QuasiIncompressiblePressureEOS(p0, rhor);
}

static bool isLinearEOS(const std::string& desc)
{
    return desc == "Linear";
}

static bool isQuasiIncompressibleEOS(const std::string& desc)
{
    return desc == "QuasiIncompressible";
}


static float readStressPeriod(ParametersWrap& desc)
{
    return desc.read<float>("stress_period");
}

template <typename PressureKernel, typename DensityKernel>
static std::shared_ptr<BasicInteractionSDPD>
allocatePairwiseSDPD(const YmrState *state, std::string name, float rc,
                     PressureKernel pressure, DensityKernel density,
                     float viscosity, float kBT,
                     bool stress, float stressPeriod)
{
    if (stress)
        return std::make_shared<InteractionSDPDWithStress<PressureKernel, DensityKernel>>
            (state, name, rc, pressure, density, viscosity, kBT, stressPeriod);
    else
        return std::make_shared<InteractionSDPD<PressureKernel, DensityKernel>>
            (state, name, rc, pressure, density, viscosity, kBT);
}

std::shared_ptr<BasicInteractionSDPD>
InteractionFactory::createPairwiseSDPD(const YmrState *state, std::string name, float rc, float viscosity, float kBT,
                                       const std::string& EOS, const std::string& density, bool stress,
                                       const MapParams& parameters)
{
    float stressPeriod = 0.f;
    ParametersWrap desc {parameters};

    if (stress)
        stressPeriod = readStressPeriod(desc);
    
    if (!isWendlandC2Density(density))
        die("Invalid density '%s'", density.c_str());

    WendlandC2DensityKernel densityKernel;
    
    if (isLinearEOS(EOS))
    {
        auto pressure = readLinearPressureEOS(desc);
        return allocatePairwiseSDPD(state, name, rc, pressure, densityKernel, viscosity, kBT, stress, stressPeriod);
    }

    if (isQuasiIncompressibleEOS(EOS))
    {
        auto pressure = readQuasiIncompressiblePressureEOS(desc);
        return allocatePairwiseSDPD(state, name, rc, pressure, densityKernel, viscosity, kBT, stress, stressPeriod);
    }

    die("Invalid pressure parameter: '%s'", EOS.c_str());
    return nullptr;
}

std::shared_ptr<InteractionDPD>
InteractionFactory::createPairwiseDPD(const YmrState *state, std::string name, float rc, float a, float gamma, float kBT, float power,
                                      bool stress, const MapParams& parameters)
{
    ParametersWrap desc {parameters};
    
    if (stress)
    {
        float stressPeriod = readStressPeriod(desc);
        return std::make_shared<InteractionDPDWithStress>(state, name, rc, a, gamma, kBT, power, stressPeriod);
    }

    return std::make_shared<InteractionDPD>(state, name, rc, a, gamma, kBT, power);
}

std::shared_ptr<InteractionMDPD>
InteractionFactory::createPairwiseMDPD(const YmrState *state, std::string name, float rc, float rd, float a, float b, float gamma, float kbt,
                                       float power, bool stress, const MapParams& parameters)
{
    ParametersWrap desc {parameters};
    
    if (stress)
    {
        float stressPeriod = readStressPeriod(desc);
        return std::make_shared<InteractionMDPDWithStress>(state, name, rc, rd, a, b, gamma, kbt, power, stressPeriod);
    }

    return std::make_shared<InteractionMDPD>(state, name, rc, rd, a, b, gamma, kbt, power);
}

std::shared_ptr<InteractionLJ>
InteractionFactory::createPairwiseLJ(const YmrState *state, std::string name, float rc, float epsilon, float sigma, float maxForce,
                                     std::string awareMode, bool stress, const MapParams& parameters)
{
    int minSegmentsDist = 0;
    ParametersWrap desc {parameters};
    
    InteractionLJ::AwareMode aMode;

    if      (awareMode == "None")   aMode = InteractionLJ::AwareMode::None;
    else if (awareMode == "Object") aMode = InteractionLJ::AwareMode::Object;
    else if (awareMode == "Rod")    aMode = InteractionLJ::AwareMode::Rod;
    else die("Invalid aware mode parameter '%s' in interaction '%s'", awareMode.c_str(), name.c_str());

    if (aMode == InteractionLJ::AwareMode::Rod)
        minSegmentsDist = (int) desc.read<float>("min_segments_distance");
    
    if (stress)
    {
        float stressPeriod = readStressPeriod(desc);
        return std::make_shared<InteractionLJWithStress>(state, name, rc, epsilon, sigma, maxForce, aMode, minSegmentsDist, stressPeriod);
    }

    return std::make_shared<InteractionLJ>(state, name, rc, epsilon, sigma, maxForce, aMode, minSegmentsDist);
}

std::shared_ptr<ObjectRodBindingInteraction>
InteractionFactory::createInteractionObjRodBinding(const YmrState *state, std::string name,
                                                   float torque, PyTypes::float3 relAnchor, float kBound)
{
    return std::make_shared<ObjectRodBindingInteraction>(state, name, torque, make_float3(relAnchor), kBound);
}
