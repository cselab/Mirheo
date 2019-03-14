#pragma once

class LinearPressureEOS
{
public:

    LinearPressureEOS(float soundSpeed) :
        c_sq(soundSpeed * soundSpeed)
    {}
    
    __D__ inline float operator()(float rho) const
    {
        return c_sq * rho;
    }

private:
    float c_sq;
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
