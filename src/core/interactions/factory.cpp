#include "factory.h"
#include "lj.h"
#include "lj_with_stress.h"
#include "membrane.h"
#include "pairwise_interactions/density_kernels.h"
#include "pairwise_interactions/pressure_EOS.h"
#include "sdpd.h"
#include "sdpd_with_stress.h"

#include <core/logger.h>

using InteractionFactory::VarParam;
using MapParams = std::map<std::string, VarParam>;


template <typename T>
static T read(const MapParams& desc, const std::string& key)
{
    auto it = desc.find(key);
    
    if (it == desc.end())
        die("missing parameter '%s'", key.c_str());

    if (!mpark::holds_alternative<T>(it->second))
        die("'%s': invalid type", key.c_str());
    
    return mpark::get<T>(it->second);
}

static CommonMembraneParameters readCommonParameters(const MapParams& desc)
{
    CommonMembraneParameters p;

    p.totArea0    = read<float>(desc, "tot_area");
    p.totVolume0  = read<float>(desc, "tot_volume");

    p.ka = read<float>(desc, "ka_tot");
    p.kv = read<float>(desc, "kv_tot");

    p.gammaC = read<float>(desc, "gammaC");
    p.gammaT = read<float>(desc, "gammaT");
    p.kBT    = read<float>(desc, "kBT");

    p.fluctuationForces = (p.kBT > 1e-6);
    
    return p;
}

static WLCParameters readWLCParameters(const MapParams& desc)
{
    WLCParameters p;

    p.x0   = read<float>(desc, "x0");
    p.ks   = read<float>(desc, "ks");
    p.mpow = read<float>(desc, "mpow");

    p.kd = read<float>(desc, "ka");
    p.totArea0 = read<float>(desc, "tot_area");
    
    return p;
}

static LimParameters readLimParameters(const MapParams& desc)
{
    LimParameters p;

    p.ka = read<float>(desc, "ka");
    p.a3 = read<float>(desc, "a3");
    p.a4 = read<float>(desc, "a4");
    
    p.mu = read<float>(desc, "mu");
    p.b1 = read<float>(desc, "b1");
    p.b2 = read<float>(desc, "b2");

    p.totArea0 = read<float>(desc, "tot_area");
    
    return p;
}

static KantorBendingParameters readKantorParameters(const MapParams& desc)
{
    KantorBendingParameters p;

    p.kb    = read<float>(desc, "kb");
    p.theta = read<float>(desc, "theta");
    
    return p;
}

static JuelicherBendingParameters readJuelicherParameters(const MapParams& desc)
{
    JuelicherBendingParameters p;

    p.kb = read<float>(desc, "kb");
    p.C0 = read<float>(desc, "C0");

    p.kad = read<float>(desc, "kad");
    p.DA0 = read<float>(desc, "DA0");
    
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
    
    auto commonPrms = readCommonParameters(parameters);

    if      (shearDesc == "wlc") shearParams = readWLCParameters(parameters);
    else if (shearDesc == "Lim") shearParams = readLimParameters(parameters);
    else                         die("No such shear parameters: '%s'", shearDesc.c_str());

    if      (bendingDesc == "Kantor")    bendingParams = readKantorParameters(parameters);
    else if (bendingDesc == "Juelicher") bendingParams = readJuelicherParameters(parameters);
    else                                 die("No such bending parameters: '%s'", bendingDesc.c_str());

    return std::make_shared<InteractionMembrane>
        (state, name, commonPrms, bendingParams, shearParams, stressFree, growUntil);
}

static RodParameters readRodParameters(const MapParams& desc)
{
    RodParameters p;
    p.kBending  = make_float3(read<PyTypes::float3>(desc, "k_bending"));
    p.omegaEq   = make_float2(read<PyTypes::float2>(desc, "omega0"));

    p.kTwist    = read<float>(desc, "k_twist");
    p.tauEq     = read<float>(desc, "tau0");
    
    p.a0 = read<float>(desc, "a0");
    p.l0 = read<float>(desc, "l0");
    p.kBounds = read<float>(desc, "k_bounds");    
    return p;
}

std::shared_ptr<InteractionRod>
InteractionFactory::createInteractionRod(const YmrState *state, std::string name,
                                         const MapParams& parameters)
{
    auto params = readRodParameters(parameters);
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


static LinearPressureEOS readLinearPressureEOS(const MapParams& desc)
{
    float c = read<float>(desc, "sound_speed");
    float r = read<float>(desc, "rho_0");
    return LinearPressureEOS(c, r);
}

static QuasiIncompressiblePressureEOS readQuasiIncompressiblePressureEOS(const MapParams& desc)
{
    float p0   = read<float>(desc, "p0");
    float rhor = read<float>(desc, "rho_r");
    
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


static float readStressPeriod(const MapParams& desc)
{
    return read<float>(desc, "stress_period");
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

    if (stress)
        stressPeriod = readStressPeriod(parameters);
    
    if (!isWendlandC2Density(density))
        die("Invalid density '%s'", density.c_str());

    WendlandC2DensityKernel densityKernel;
    
    if (isLinearEOS(EOS))
    {
        auto pressure = readLinearPressureEOS(parameters);
        return allocatePairwiseSDPD(state, name, rc, pressure, densityKernel, viscosity, kBT, stress, stressPeriod);
    }

    if (isQuasiIncompressibleEOS(EOS))
    {
        auto pressure = readQuasiIncompressiblePressureEOS(parameters);
        return allocatePairwiseSDPD(state, name, rc, pressure, densityKernel, viscosity, kBT, stress, stressPeriod);
    }

    die("Invalid pressure parameter: '%s'", EOS.c_str());
    return nullptr;
}

std::shared_ptr<InteractionLJ>
InteractionFactory::createPairwiseLJ(const YmrState *state, std::string name, float rc, float epsilon, float sigma, float maxForce,
                                     std::string awareMode, bool stress, const MapParams& parameters)
{
    int minSegmentsDist = 0;

    InteractionLJ::AwareMode aMode;

    if      (awareMode == "None")   aMode = InteractionLJ::AwareMode::None;
    else if (awareMode == "Object") aMode = InteractionLJ::AwareMode::Object;
    else if (awareMode == "Rod")    aMode = InteractionLJ::AwareMode::Rod;
    else die("Invalid aware mode parameter '%s' in interaction '%s'", awareMode.c_str(), name.c_str());

    if (aMode == InteractionLJ::AwareMode::Rod)
        minSegmentsDist = (int) read<float>(parameters, "min_segments_distance");
    
    if (stress)
    {
        float stressPeriod = readStressPeriod(parameters);
        return std::make_shared<InteractionLJWithStress>(state, name, rc, epsilon, sigma, maxForce, aMode, minSegmentsDist, stressPeriod);
    }

    return std::make_shared<InteractionLJ>(state, name, rc, epsilon, sigma, maxForce, aMode, minSegmentsDist);
}
