#pragma once

#include <cmath>

class SimpleMPDPDDendityKernel
{
public:
    static constexpr float normalization = 15 / (2 * M_PI);

    __D__ inline float operator()(float r, float inv_rc) const
    {
        float rm = (1.f - r * inv_rc) * inv_rc;

        return normalization * rm * rm * inv_rc;
    }
};

class WendlandC2DensityKernel
{
public:
    static constexpr float normalization = 21 / (2 * M_PI);
    
    __D__ inline float operator()(float r, float inv_rc) const
    {
        float r_ = r * inv_rc;
        float rm = 1.f - r_;
        float rm2 = rm * rm;
        
        return normalization * rm2 * rm2 * (1 + 4 * r_);
    }

    __D__ inline float derivative(float r, float inv_rc) const
    {
        float r_ = r * inv_rc;
        float rm = r_ - 1.f;
        return normalization * 4 * r_ * rm*rm*rm * inv_rc;
    }
};
