#include "factory_helper.h"

namespace FactoryHelper
{
template <> float ParamsReader::makeDefault<float>() const {return defaultFloat;}

template <> void readParams<DPDParams>(DPDParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto a     = reader.read<float>(desc, "a");
    const auto gamma = reader.read<float>(desc, "gamma");
    const auto kBT   = reader.read<float>(desc, "kbt");
    const auto power = reader.read<float>(desc, "power");
    
    if (a     != defaultFloat) p.a     = a;
    if (gamma != defaultFloat) p.gamma = gamma;
    if (kBT   != defaultFloat) p.kBT   = kBT;
    if (power != defaultFloat) p.power = power;
}

template <> void readParams<LJAwarenessParamsNone  >(__UNUSED LJAwarenessParamsNone&   p, __UNUSED ParametersWrap& desc, __UNUSED ParamsReader reader) {}
template <> void readParams<LJAwarenessParamsObject>(__UNUSED LJAwarenessParamsObject& p, __UNUSED ParametersWrap& desc, __UNUSED ParamsReader reader) {}
template <> void readParams<LJAwarenessParamsRod>(LJAwarenessParamsRod& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto minSegmentsDist = reader.read<float>(desc, "min_segments_distance");
    if (minSegmentsDist != defaultFloat) p.minSegmentsDist = static_cast<int>(minSegmentsDist);
}

template <> void readParams<LJParams>(LJParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto epsilon  = reader.read<float>(desc, "epsilon");
    const auto sigma    = reader.read<float>(desc, "sigma");
    const auto maxForce = reader.read<float>(desc, "max_force");

    if (epsilon  != defaultFloat) p.epsilon  = epsilon;
    if (sigma    != defaultFloat) p.sigma    = sigma;
    if (maxForce != defaultFloat) p.maxForce = maxForce;
}

template <> void readParams<MDPDParams>(MDPDParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto rd    = reader.read<float>(desc, "rd");
    const auto a     = reader.read<float>(desc, "a");
    const auto b     = reader.read<float>(desc, "b");
    const auto gamma = reader.read<float>(desc, "gamma");
    const auto kBT   = reader.read<float>(desc, "kbt");
    const auto power = reader.read<float>(desc, "power");
    
    if (rd     != defaultFloat) p.rd    = rd;
    if (a      != defaultFloat) p.a     = a;
    if (b      != defaultFloat) p.b     = b;
    if (gamma  != defaultFloat) p.gamma = gamma;
    if (kBT    != defaultFloat) p.kBT   = kBT;
    if (power  != defaultFloat) p.power = power;
}

template <> void readParams<DensityParams>(__UNUSED DensityParams& p, __UNUSED ParametersWrap& desc, __UNUSED ParamsReader reader) {}

template <> void readParams<SimpleMDPDDensityKernelParams>(__UNUSED SimpleMDPDDensityKernelParams& p, __UNUSED ParametersWrap& desc, __UNUSED ParamsReader reader) {}
template <> void readParams<WendlandC2DensityKernelParams>(__UNUSED WendlandC2DensityKernelParams& p, __UNUSED ParametersWrap& desc, __UNUSED ParamsReader reader) {}

template <> void readParams<LinearPressureEOSParams>(LinearPressureEOSParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto soundSpeed = reader.read<float>(desc, "sound_speed");
    const auto rho0       = reader.read<float>(desc, "rho_0");

    if (soundSpeed != defaultFloat) p.soundSpeed = soundSpeed;
    if (rho0       != defaultFloat) p.rho0       = rho0;
}

template <> void readParams<QuasiIncompressiblePressureEOSParams>(QuasiIncompressiblePressureEOSParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto p0   = reader.read<float>(desc, "p0");
    const auto rhor = reader.read<float>(desc, "rho_r");

    if (p0   != defaultFloat) p.p0   = p0;
    if (rhor != defaultFloat) p.rhor = rhor;
}

template <> void readParams<SDPDParams>(SDPDParams& p, ParametersWrap& desc, ParamsReader reader)
{
    const auto viscosity = reader.read<float>(desc, "viscosity");
    const auto kBT       = reader.read<float>(desc, "kBT");

    if (viscosity != defaultFloat) p.viscosity = viscosity;
    if (kBT       != defaultFloat) p.kBT       = kBT;
}



DPDParams readDPDParams(ParametersWrap& desc)
{
    DPDParams p;
    readParams(p, desc, {ParamsReader::Mode::FailIfNotFound});
    return p;
}

static VarLJAwarenessParams readLJAwarenessParams(ParametersWrap& desc, ParamsReader reader)
{
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
    const auto kernel = desc.read<std::string>("kernel");
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
        const auto period = desc.read<float>("stress_period");
        return StressActiveParams {period};
    }
    else
    {
        return StressNoneParams {};
    }
}

} // namespace FactoryHelper
