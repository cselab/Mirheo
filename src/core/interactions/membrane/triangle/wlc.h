#pragma once

#include "../parameters.h"

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>
#include <core/mesh/membrane.h>

#include <cmath>


enum class StressFreeState
{
    Active,
    Inactive
};


template <StressFreeState stressFreeState>
class TriangleWLCForce
{
public:    
    struct LengthArea
    {
        float l0, a0;
    };

    using EquilibriumTriangleDesc = LengthArea;
    using ParametersType          = WLCParameters;
    
    TriangleWLCForce(ParametersType p, const Mesh *mesh, float lscale) :
        lscale(lscale)
    {
        x0   = p.x0;
        ks   = p.ks * lscale * lscale;
        mpow = p.mpow;

        kd = p.kd * lscale * lscale;
        
        area0   = p.totArea0 * lscale * lscale / mesh->getNtriangles();
        length0 = sqrt(area0 * 4.0 / sqrt(3.0));
    }

    __D__ inline void initEquilibriumDesc(const MembraneMeshView& mesh, int startId) const
    {}
    
    __D__ inline EquilibriumTriangleDesc getEquilibriumDesc(const MembraneMeshView& mesh, int i) const
    {
        LengthArea eq;
        if (stressFreeState == StressFreeState::Active)
        {
            eq.l0 = mesh.initialLengths[i] * lscale;
            eq.a0 = mesh.initialAreas  [i] * lscale;
        }
        else
        {
            eq.l0 = this->length0;
            eq.a0 = this->area0;
        }
        return eq;
    }

    __D__ inline float3 operator()(float3 v1, float3 v2, float3 v3, EquilibriumTriangleDesc eq) const
    {
        return areaForce(v1, v2, v3, eq.a0) + bondForce(v1, v2, eq.l0);
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

    __D__ inline float3 areaForce(float3 v1, float3 v2, float3 v3, float area0) const
    {
        float3 x21 = v2 - v1;
        float3 x32 = v3 - v2;
        float3 x31 = v3 - v1;

        float3 normal = cross(x21, x31);

        float area = 0.5f * length(normal);

        float coef = kd * (area - area0) / (area * area0);

        return -0.25f * coef * cross(normal, x32);
    }


    static constexpr float forceCap = 1500.f;
    float x0, ks, mpow;
    float kd;

    float length0, area0; ///< only useful when StressFree is false
    float lscale;
};
