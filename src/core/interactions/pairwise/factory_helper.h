#pragma once

#include "kernels/parameters.h"

#include <core/interactions/parameters_wrap.h>

#include <limits>

constexpr auto defaultFloat = std::numeric_limits<float>::infinity();

DPDParams readDPDParams(ParametersWrap& desc)
{
    DPDParams p;
    p.a     = desc.read<float>("a");
    p.gamma = desc.read<float>("gamma");
    p.kBT   = desc.read<float>("kbt");
    p.power = desc.read<float>("power");
    return p;
}

VarLJAwarenessParams readLJAwarenessParams(ParametersWrap& desc)
{
    VarLJAwarenessParams p;
    const auto awareMode = desc.read<std::string>("aware_mode");

    if (awareMode == "None")
    {
        p = LJAwarenessParamsNone {};
    }
    else if (awareMode == "Object")
    {
        p = LJAwarenessParamsObject {};
    }
    else if (awareMode == "Rod")
    {
        const auto minSegmentsDist = static_cast<int>(desc.read<float>("min_segments_distance"));
        LJAwarenessParamsRod {minSegmentsDist};
    }
    else
    {
        die("Unrecognized aware mode '%s'", awareMode.c_str());
    }
    
    return p;
}

LJParams readLJParams(ParametersWrap& desc)
{
    LJParams p;
    p.epsilon  = desc.read<float>("epsilon");
    p.sigma    = desc.read<float>("sigma");
    p.maxForce = desc.read<float>("max_force");
    p.varLJAwarenessParams = readLJAwarenessParams(desc);
    return p;
}


MDPDParams readMDPDParams(ParametersWrap& desc)
{
    MDPDParams p;
    p.rd    = desc.read<float>("rd");
    p.a     = desc.read<float>("a");
    p.b     = desc.read<float>("b");
    p.gamma = desc.read<float>("gamma");
    p.kBT   = desc.read<float>("kbt");
    p.power = desc.read<float>("power");
    return p;
}

DensityParams readDensityParams(ParametersWrap& desc)
{
    DensityParams p;
    const auto kernel = desc.read<std::string>("kernel");
    if (kernel == "MDPD")
        p.varDensityKernelParams = SimpleMDPDDensityKernelParams {};
    else if (kernel == "WendlandC2")
        p.varDensityKernelParams = WendlandC2DensityKernelParams {};
    else
        die("unrecognized density kernel '%d'", kernel.c_str());
    return p;
}

VarSDPDDensityKernelParams readSDPDDensityKernelParams(ParametersWrap& desc)
{
    VarSDPDDensityKernelParams p;
    const auto kernel = desc.read<std::string>("density_kernel");
    if (kernel == "WendlandC2")
        p = WendlandC2DensityKernelParams {};
    else
        die("unrecognized density kernel '%d'", kernel.c_str());
    return p;
}

VarEOSParams readEOSParams(ParametersWrap& desc)
{
    VarEOSParams varEOS;
    const auto eos = desc.read<std::string>("EOS");

    if (eos == "Linear")
    {
        const auto soundSpeed = desc.read<float>("sound_speed");
        const auto rho0       = desc.read<float>("rho_0");
        varEOS = LinearPressureEOSParams {soundSpeed, rho0};
    }
    else if (eos == "QuasiIncompressible")
    {
        const auto p0   = desc.read<float>("p0");
        const auto rhor = desc.read<float>("rho_r");
        varEOS = LinearPressureEOSParams {p0, rhor};
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

    p.viscosity = desc.read<float>("viscosity");
    p.kBT       = desc.read<float>("kBT");

    p.varEOSParams           = readEOSParams(desc);
    p.varDensityKernelParams = readSDPDDensityKernelParams(desc);

    return p;
}
