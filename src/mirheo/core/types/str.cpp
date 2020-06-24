// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "str.h"

#include <mirheo/core/utils/strprintf.h>

namespace mirheo
{

std::string printToStr(int val)
{
    return strprintf("%d", val);
}

std::string printToStr(int64_t val)
{
    return strprintf("%ld", val);
}

std::string printToStr(float val)
{
    return strprintf("%g", val);
}

std::string printToStr(float2 val)
{
    return strprintf("%g %g", val.x, val.y);
}

std::string printToStr(float3 val)
{
    return strprintf("%g %g %g", val.x, val.y, val.z);
}

std::string printToStr(float4 val)
{
    return strprintf("%g %g %g %g", val.x, val.y, val.z, val.w);
}


std::string printToStr(double val)
{
    return strprintf("%g", val);
}

std::string printToStr(double2 val)
{
    return strprintf("%g %g", val.x, val.y);
}

std::string printToStr(double3 val)
{
    return strprintf("%g %g %g", val.x, val.y, val.z);
}

std::string printToStr(double4 val)
{
    return strprintf("%g %g %g %g", val.x, val.y, val.z, val.w);
}

std::string printToStr(Stress val)
{
    return strprintf("%g %g %g %g %g %g",
                     val.xx, val.xy, val.xz,
                     val.yy, val.yz, val.zz);
}


std::string printToStr(RigidMotion val)
{
    return strprintf("[r: %s, q: %g %g %g %g, v: %s, w: %s, F: %s, T: %s]",
                     printToStr(val.r).c_str(),
                     val.q.w, val.q.x, val.q.y, val.q.z,
                     printToStr(val.vel).c_str(),
                     printToStr(val.omega).c_str(),
                     printToStr(val.force).c_str(),
                     printToStr(val.torque).c_str());
}

std::string printToStr(COMandExtent val)
{
    return strprintf("[com: %s, lo: %s, hi: %s]",
                     printToStr(val.com).c_str(),
                     printToStr(val.low).c_str(),
                     printToStr(val.high).c_str());
}

std::string printToStr(Force val)
{
    return printToStr(val.f);
}

} // namespace mirheo
