#pragma once

#include "local_area.h"

class TriangleWLCForce : public LocalAreaForce
{
public:    

    using ParametersType = WLCParameters;
    
    TriangleWLCForce(ParametersType p, float lscale) :
        LocalAreaForce(p.kd, lscale)
    {
        x0   = p.x0;
        ks   = p.ks * lscale * lscale;
        mpow = p.mpow;
    }

    __D__ inline float3 operator()(float3 v1, float3 v2, float3 v3, float l0, float a0) const
    {
        return LocalAreaForce::areaForce(v1, v2, v3, a0) + bondForce(v1, v2, l0);
    }
        
private:

    __D__ inline float3 bondForce(float3 v1, float3 v2, float l0) const
    {
        float r = max(length(v2 - v1), 1e-5f);
        float lmax     = l0 / x0;
        float inv_lmax = x0 / l0;

        auto wlc = [this, inv_lmax] (float x) {
            return ks * inv_lmax * (4.0f*x*x - 9.0f*x + 6.0f) / ( 4.0f*sqr(1.0f - x) );
        };

        float IbforceI_wlc = wlc( min(lmax - 1e-6f, r) * inv_lmax );

        float kp = wlc( l0 * inv_lmax ) * fastPower(l0, mpow+1);

        float IbforceI_pow = -kp / (fastPower(r, mpow+1));

        float IfI = min(forceCap, max(-forceCap, IbforceI_wlc + IbforceI_pow));

        return IfI * (v2 - v1);
    }

    static constexpr float forceCap = 1500.f;
    float x0, ks, mpow;
};
