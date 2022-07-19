// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory_helper.h"

namespace mirheo {
namespace factory_helper {


DPDParams readDPDParams(ParametersWrap& desc)
{
    DPDParams p;
    p.a     = desc.read<real>("a");
    p.gamma = desc.read<real>("gamma");
    p.kBT   = desc.read<real>("kBT");
    p.power = desc.read<real>("power");
    return p;
}

static VarAwarenessParams readAwarenessParams(ParametersWrap& desc)
{
    if (!desc.exists<std::string>("aware_mode"))
        return AwarenessParamsNone {};

    VarAwarenessParams varP;

    const auto awareMode = desc.read<std::string>("aware_mode");

    if (awareMode == "None")
    {
        varP = AwarenessParamsNone{};
    }
    else if (awareMode == "Object")
    {
        varP = AwarenessParamsObject{};
    }
    else if (awareMode == "Rod")
    {
        AwarenessParamsRod p;
        p.minSegmentsDist = static_cast<int>(desc.read<real>("min_segments_distance"));
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
    LJParams p;
    p.epsilon = desc.read<real>("epsilon");
    p.sigma   = desc.read<real>("sigma");
    return p;
}

RepulsiveLJParams readRepulsiveLJParams(ParametersWrap& desc)
{
    RepulsiveLJParams p;
    p.epsilon  = desc.read<real>("epsilon");
    p.sigma    = desc.read<real>("sigma");
    p.maxForce = desc.read<real>("max_force");
    p.varAwarenessParams = readAwarenessParams(desc);
    return p;
}

GrowingRepulsiveLJParams readGrowingRepulsiveLJParams(ParametersWrap& desc)
{
    GrowingRepulsiveLJParams p;
    p.epsilon  = desc.read<real>("epsilon");
    p.sigma    = desc.read<real>("sigma");
    p.maxForce = desc.read<real>("max_force");
    p.initialLengthFraction = desc.read<real>("init_length_fraction");
    p.growUntil             = desc.read<real>("grow_until");
    p.varAwarenessParams = readAwarenessParams(desc);
    return p;
}

MorseParams readMorseParams(ParametersWrap& desc)
{
    MorseParams p;
    p.De   = desc.read<real>("De");
    p.r0   = desc.read<real>("r0");
    p.beta = desc.read<real>("beta");
    p.varAwarenessParams = readAwarenessParams(desc);
    return p;
}

MDPDParams readMDPDParams(ParametersWrap& desc)
{
    MDPDParams p;
    p.rd    = desc.read<real>("rd");
    p.a     = desc.read<real>("a");
    p.b     = desc.read<real>("b");
    p.gamma = desc.read<real>("gamma");
    p.kBT   = desc.read<real>("kBT");
    p.power = desc.read<real>("power");
    return p;
}

DensityParams readDensityParams(ParametersWrap& desc)
{
    DensityParams p;
    const auto kernel = desc.read<std::string>("density_kernel");

    if (kernel == "MDPD")
    {
        p.varDensityKernelParams = SimpleMDPDDensityKernelParams{};
    }
    else if (kernel == "WendlandC2")
    {
        p.varDensityKernelParams = WendlandC2DensityKernelParams{};
    }
    else
    {
        die("unrecognized density kernel '%s'", kernel.c_str());
    }
    return p;
}

static VarSDPDDensityKernelParams readSDPDDensityKernelParams(ParametersWrap& desc)
{
    VarSDPDDensityKernelParams p;
    const auto kernel = desc.read<std::string>("density_kernel");

    if (kernel == "WendlandC2")
    {
        p = WendlandC2DensityKernelParams{};
    }
    else
    {
        die("unrecognized density kernel '%s'", kernel.c_str());
    }
    return p;
}

static VarEOSParams readEOSParams(ParametersWrap& desc)
{
    VarEOSParams varEOS;
    const auto eos = desc.read<std::string>("EOS");

    if (eos == "Linear")
    {
        LinearPressureEOSParams p;
        p.soundSpeed = desc.read<real>("sound_speed");
        p.rho0       = desc.read<real>("rho_0");
        varEOS = p;
    }
    else if (eos == "QuasiIncompressible")
    {
        QuasiIncompressiblePressureEOSParams p;
        p.p0   = desc.read<real>("p0");
        p.rhor = desc.read<real>("rho_r");
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
    SDPDParams p;
    p.viscosity = desc.read<real>("viscosity");
    p.kBT       = desc.read<real>("kBT");
    p.varEOSParams           = readEOSParams(desc);
    p.varDensityKernelParams = readSDPDDensityKernelParams(desc);
    return p;
}


VarStressParams readStressParams(ParametersWrap& desc)
{
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

std::optional<real> readStressPeriod(ParametersWrap& desc)
{
    bool stress {false};

    if (desc.exists<bool>("stress"))
        stress = desc.read<bool>("stress");

    if (stress)
    {
        const auto period = desc.read<real>("stress_period");
        return {period};
    }
    else
    {
        return std::nullopt;
    }

}

} // namespace factory_helper
} // namespace mirheo
