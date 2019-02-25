#include "factory.h"

#include "membrane/parameters.h"

#include "membrane_WLC_Kantor.h"
#include "membrane_WLC_Juelicher.h"

#include <core/logger.h>

static bool hasKey(const std::map<std::string, float>& desc, const std::string& key)
{
    return desc.find(key) != desc.end();
}

static float readFloat(const std::map<std::string, float>& desc, const std::string& key)
{
    auto it = desc.find(key);
    
    if (it == desc.end())
        die("missing parameter '%s'", key.c_str());
    
    return it->second;
}

static MembraneParameters readCommonParameters(const std::map<std::string, float>& desc)
{
    MembraneParameters p;

    p.totArea0    = readFloat(desc, "tot_area");
    p.totVolume0  = readFloat(desc, "tot_volume");

    p.ka = readFloat(desc, "ka_tot");
    p.kv = readFloat(desc, "kv_tot");

    p.gammaC = readFloat(desc, "gammaC");
    p.gammaT = readFloat(desc, "gammaT");
    p.kBT    = readFloat(desc, "kBT");

    p.fluctuationForces = (p.kBT > 1e-6);
    
    return p;
}

static WLCParameters readWLCParameters(const std::map<std::string, float>& desc)
{
    WLCParameters p;

    p.x0   = readFloat(desc, "x0");
    p.ks   = readFloat(desc, "ks");
    p.mpow = readFloat(desc, "mpow");

    p.kd = readFloat(desc, "ka");
    p.totArea0 = readFloat(desc, "tot_area");
    
    return p;
}

static LimParameters readLimParameters(const std::map<std::string, float>& desc)
{
    LimParameters p;

    p.ka = readFloat(desc, "ka");
    p.a3 = readFloat(desc, "a3");
    p.a4 = readFloat(desc, "a4");
    
    p.mu = readFloat(desc, "mu");
    p.b1 = readFloat(desc, "b1");
    p.b2 = readFloat(desc, "b2");

    p.totArea0 = readFloat(desc, "tot_area");
    
    return p;
}

static KantorBendingParameters readKantorParameters(const std::map<std::string, float>& desc)
{
    KantorBendingParameters p;

    p.kb    = readFloat(desc, "kb");
    p.theta = readFloat(desc, "ks");
    
    return p;
}

static JuelicherBendingParameters readJuelicherParameters(const std::map<std::string, float>& desc)
{
    JuelicherBendingParameters p;

    p.kb = readFloat(desc, "kb");
    p.C0 = readFloat(desc, "C0");

    p.kad = readFloat(desc, "kad");
    p.DA0 = readFloat(desc, "DA0");
    
    return p;
}

static bool isWLC(const std::string& desc)
{
    return desc == "wlc";
}

static bool isLim(const std::string& desc)
{
    return desc == "Lim";
}

static bool isKantor(const std::string& desc)
{
    return desc == "Kantor";
}

static bool isJuelicher(const std::string& desc)
{
    return desc == "Juelicher";
}


std::shared_ptr<InteractionMembrane>
InteractionFactory::createInteractionMembrane(const YmrState *state, std::string name,
                                              std::string shearDesc, std::string bendingDesc,
                                              const std::map<std::string, float>& parameters,
                                              bool stressFree, float growUntil)
{
    auto commonPrms = readCommonParameters(parameters);
    
    if (isWLC(shearDesc))
    {
        auto shPrms = readWLCParameters(parameters);
            
        if (isKantor(bendingDesc))
        {
            auto bePrms = readKantorParameters(parameters);
            return std::make_shared<InteractionMembraneWLCKantor>
                (state, name, commonPrms, shPrms, bePrms, stressFree, growUntil);
        }

        if (isJuelicher(bendingDesc))
        {
            auto bePrms = readJuelicherParameters(parameters);
            return std::make_shared<InteractionMembraneWLCJuelicher>
                (state, name, commonPrms, shPrms, bePrms, stressFree, growUntil);
        }            
    }

    die("argument combination of shearDesc = '%s' and bendingDesc = '%s' is incorrect",
        shearDesc.c_str(), bendingDesc.c_str());

    return nullptr;
}
    
