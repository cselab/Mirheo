#pragma once

#include <core/utils/cpu_gpu_defines.h>

class LinearPressureEOS
{
public:

    LinearPressureEOS(float soundSpeed, float rho0) :
        cSq(soundSpeed * soundSpeed)
    {}
    
    __D__ inline float operator()(float rho) const
    {
        return cSq * (rho - rho0);
    }

private:
    float cSq, rho0;
};


class QuasiIncompressiblePressureEOS
{
public:
    
    QuasiIncompressiblePressureEOS(float p0, float rhor) :
        p0(p0),
        rhor(rhor)
    {}

    __D__ inline float operator()(float rho) const
    {
        float r = rho / rhor;
        float r3 = r*r*r;
        float r7 = r3*r3*r;
        return p0 * (r7 - 1.f);
    }

private:

    float p0, rhor;
};
