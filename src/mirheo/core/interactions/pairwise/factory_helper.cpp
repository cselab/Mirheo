#include "factory_helper.h"

namespace FactoryHelper
{
template <> real ParamsReader::makeDefault<real>() const {return defaultReal;}

template <> void readParams<DPDParams>(DPDParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto a     = reader.read<real>(desc, "a");
    const auto gamma = reader.read<real>(desc, "gamma");
    const auto kBT   = reader.read<real>(desc, "kBT");
    const auto power = reader.read<real>(desc, "power");
    
    if (a     != defaultReal) p.a     = a;
    if (gamma != defaultReal) p.gamma = gamma;
    if (kBT   != defaultReal) p.kBT   = kBT;
    if (power != defaultReal) p.power = power;
}

template <> void readParams<LJAwarenessParamsNone  >(__UNUSED LJAwarenessParamsNone&   p, __UNUSED ParametersWrap& desc, __UNUSED ParamsReader reader) {}
template <> void readParams<LJAwarenessParamsObject>(__UNUSED LJAwarenessParamsObject& p, __UNUSED ParametersWrap& desc, __UNUSED ParamsReader reader) {}
template <> void readParams<LJAwarenessParamsRod>(LJAwarenessParamsRod& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto minSegmentsDist = reader.read<real>(desc, "min_segments_distance");
    if (minSegmentsDist != defaultReal) p.minSegmentsDist = static_cast<int>(minSegmentsDist);
}

template <> void readParams<LJParams>(LJParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto epsilon  = reader.read<real>(desc, "epsilon");
    const auto sigma    = reader.read<real>(desc, "sigma");
    const auto maxForce = reader.read<real>(desc, "max_force");

    if (epsilon  != defaultReal) p.epsilon  = epsilon;
    if (sigma    != defaultReal) p.sigma    = sigma;
    if (maxForce != defaultReal) p.maxForce = maxForce;
}

template <> void readParams<MDPDParams>(MDPDParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto rd    = reader.read<real>(desc, "rd");
    const auto a     = reader.read<real>(desc, "a");
    const auto b     = reader.read<real>(desc, "b");
    const auto gamma = reader.read<real>(desc, "gamma");
    const auto kBT   = reader.read<real>(desc, "kBT");
    const auto power = reader.read<real>(desc, "power");
    
    if (rd     != defaultReal) p.rd    = rd;
    if (a      != defaultReal) p.a     = a;
    if (b      != defaultReal) p.b     = b;
    if (gamma  != defaultReal) p.gamma = gamma;
    if (kBT    != defaultReal) p.kBT   = kBT;
    if (power  != defaultReal) p.power = power;
}

template <> void readParams<DensityParams>(__UNUSED DensityParams& p, __UNUSED ParametersWrap& desc, __UNUSED ParamsReader reader) {}

template <> void readParams<SimpleMDPDDensityKernelParams>(__UNUSED SimpleMDPDDensityKernelParams& p, __UNUSED ParametersWrap& desc, __UNUSED ParamsReader reader) {}
template <> void readParams<WendlandC2DensityKernelParams>(__UNUSED WendlandC2DensityKernelParams& p, __UNUSED ParametersWrap& desc, __UNUSED ParamsReader reader) {}

template <> void readParams<LinearPressureEOSParams>(LinearPressureEOSParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto soundSpeed = reader.read<real>(desc, "sound_speed");
    const auto rho0       = reader.read<real>(desc, "rho_0");

    if (soundSpeed != defaultReal) p.soundSpeed = soundSpeed;
    if (rho0       != defaultReal) p.rho0       = rho0;
}

template <> void readParams<QuasiIncompressiblePressureEOSParams>(QuasiIncompressiblePressureEOSParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto p0   = reader.read<real>(desc, "p0");
    const auto rhor = reader.read<real>(desc, "rho_r");

    if (p0   != defaultReal) p.p0   = p0;
    if (rhor != defaultReal) p.rhor = rhor;
}

template <> void readParams<SDPDParams>(SDPDParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto viscosity = reader.read<real>(desc, "viscosity");
    const auto kBT       = reader.read<real>(desc, "kBT");

    if (viscosity != defaultReal) p.viscosity = viscosity;
    if (kBT       != defaultReal) p.kBT       = kBT;
}



DPDParams readDPDParams(ParametersWrap& desc)
{
    DPDParams p;
    readParams(p, desc, {ParamsReader::Mode::FailIfNotFound});
    return p;
}

static VarLJAwarenessParams readLJAwarenessParams(ParametersWrap& desc, ParamsReader reader)
{
    if (!desc.exists<std::string>("aware_mode"))
        return LJAwarenessParamsNone {};
    
    VarLJAwarenessParams varP;

    const auto awareMode = desc.read<std::string>("aware_mode");

    if (awareMode == "None")
    {
        LJAwarenessParamsNone p;
        readParams(p, desc, reader);
        varP = p;
    }
    else if (awareMode == "Object")
    {
        LJAwarenessParamsObject p;
        readParams(p, desc, reader);
        varP = p;
    }
    else if (awareMode == "Rod")
    {
        LJAwarenessParamsRod p;
        readParams(p, desc, reader);
        varP = p;
    }
    else
    {
        die("Unrecognized aware mode '%s'", awareMode.c_str());
    }
    
    return varP;
}

LJParams readLJParams(ParametersWrap& desc)
{
    const ParamsReader reader {ParamsReader::Mode::FailIfNotFound};
    LJParams p;
    readParams(p, desc, reader);
    p.varLJAwarenessParams = readLJAwarenessParams(desc, reader);
    return p;
}


MDPDParams readMDPDParams(ParametersWrap& desc)
{
    MDPDParams p;
    readParams(p, desc, {ParamsReader::Mode::FailIfNotFound});
    return p;
}

DensityParams readDensityParams(ParametersWrap& desc)
{
    DensityParams p;
    const auto kernel = desc.read<std::string>("density_kernel");
    const ParamsReader reader {ParamsReader::Mode::FailIfNotFound};

    if (kernel == "MDPD")
    {
        SimpleMDPDDensityKernelParams density;
        readParams(density, desc, reader);
        p.varDensityKernelParams = density;
    }
    else if (kernel == "WendlandC2")
    {
        WendlandC2DensityKernelParams density;
        readParams(density, desc, reader);
        p.varDensityKernelParams = density;
    }
    else
    {
        die("unrecognized density kernel '%d'", kernel.c_str());
    }
    return p;
}

static VarSDPDDensityKernelParams readSDPDDensityKernelParams(ParametersWrap& desc, ParamsReader reader)
{
    VarSDPDDensityKernelParams p;
    const auto kernel = desc.read<std::string>("density_kernel");
    
    if (kernel == "WendlandC2")
    {
        WendlandC2DensityKernelParams density;
        readParams(density, desc, reader);
        p = density;
    }
    else
    {
        die("unrecognized density kernel '%d'", kernel.c_str());
    }
    return p;
}

static VarEOSParams readEOSParams(ParametersWrap& desc, ParamsReader reader)
{
    VarEOSParams varEOS;
    const auto eos = desc.read<std::string>("EOS");
    
    if (eos == "Linear")
    {
        LinearPressureEOSParams p;
        readParams(p, desc, reader);
        varEOS = p;
    }
    else if (eos == "QuasiIncompressible")
    {
        QuasiIncompressiblePressureEOSParams p;
        readParams(p, desc, reader);
        varEOS = p;
    }
    else
    {
        die("Unrecognizes equation of state '%s'", eos.c_str());
    }
    return varEOS;
}

SDPDParams readSDPDParams(ParametersWrap& desc)
{
    const ParamsReader reader {ParamsReader::Mode::FailIfNotFound};
    SDPDParams p;

    readParams(p, desc, reader);

    p.varEOSParams           = readEOSParams(desc, reader);
    p.varDensityKernelParams = readSDPDDensityKernelParams(desc, reader);

    return p;
}



VarStressParams readStressParams(ParametersWrap& desc)
{
    VarStressParams varParams;
    bool stress {false};

    if (desc.exists<bool>("stress"))
        stress = desc.read<bool>("stress");

    if (stress)
    {
        const auto period = desc.read<real>("stress_period");
        return StressActiveParams {period};
    }
    else
    {
        return StressNoneParams {};
    }
}

} // namespace FactoryHelper
