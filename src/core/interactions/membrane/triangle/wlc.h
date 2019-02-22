#pragma once

#include "triangle_force.h"

class TriangleWLC : public TriangleForce
{
public:    

    using ParametersType = WLCParameters;
    
    TriangleWLC(float ka, float kd, float totArea0, ParametersType p, float lscale) :
        TriangleForce(ka, kd, totArea0, lscale)
    {
        x0   = p.x0;
        ks   = p.ks * lscale * lscale;
        mpow = p.mpow;
    }

    __D__ inline float3 bond(VertexType v0, VertexType v1, float l0) const
    {
        float r = max(length(v2 - v1), 1e-5f);
        float lmax     = l0 / parameters.x0;
        float inv_lmax = parameters.x0 / l0;

        auto wlc = [parameters, inv_lmax] (float x) {
                       return parameters.ks * inv_lmax * (4.0f*x*x - 9.0f*x + 6.0f) / ( 4.0f*sqr(1.0f - x) );
                   };

        float IbforceI_wlc = wlc( min(lmax - 1e-6f, r) * inv_lmax );

        float kp = wlc( l0 * inv_lmax ) * fastPower(l0, parameters.mpow+1);

        float IbforceI_pow = -kp / (fastPower(r, parameters.mpow+1));

        float IfI = min(forceCap, max(-forceCap, IbforceI_wlc + IbforceI_pow));

        return IfI * (v2 - v1);
    }

    using TriangleForce::areaForce;
        
private:

    float x0, ks, mpow;
};
