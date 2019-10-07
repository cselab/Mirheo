#pragma once

#include <cmath>

class SimpleMDPDDensityKernel
{
public:
    static constexpr float normalization = 15.0 / (2.0 * M_PI);

    __D__ inline float operator()(float r, float invrc) const
    {
        const float rm = (1.f - r * invrc) * invrc;

        return normalization * rm * rm * invrc;
    }

    __D__ inline float derivative(float r, float invrc) const
    {
        const float rm = 1.f - r * invrc;
        const float invrc2 = invrc  * invrc;
        const float invrc4 = invrc2 * invrc2;
        return 2.0f * normalization * rm * invrc4;
    }
};

class WendlandC2DensityKernel
{
public:
    static constexpr float normalization = 21.0 / (2.0 * M_PI);

    __D__ inline float operator()(float r, float invrc) const
    {
        const float r_ = r * invrc;
        const float rm = 1.0f - r_;
        const float rm2 = rm * rm;
        
        return normalization * rm2 * rm2 * (1.0f + 4.0f * r_);
    }

    __D__ inline float derivative(float r, float invrc) const
    {
        const float r_ = r * invrc;
        const float rm = r_ - 1.f;
        return 20.0f * normalization * r_ * rm*rm*rm * invrc;
    }
};
