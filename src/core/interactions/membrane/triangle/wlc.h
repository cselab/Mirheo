#pragma once

#include "../parameters.h"

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>
#include <core/mesh/membrane.h>

#include <cmath>

template <StressFreeState stressFreeState>
class TriangleWLCForce
{
public:    
    struct LengthArea
    {
        real l; // eq. edge length
        real a; // eq. triangle area
    };

    using EquilibriumTriangleDesc = LengthArea;
    using ParametersType          = WLCParameters;
    
    TriangleWLCForce(ParametersType p, const Mesh *mesh, real lscale) :
        lscale(lscale)
    {
        x0   = p.x0;
        ks   = p.ks * lscale * lscale;
        mpow = p.mpow;

        kd = p.kd * lscale * lscale;
        
        area0   = p.totArea0 * lscale * lscale / mesh->getNtriangles();
        length0 = sqrt(area0 * 4.0 / sqrt(3.0));
    }

    __D__ inline EquilibriumTriangleDesc getEquilibriumDesc(const MembraneMeshView& mesh, int i0, int i1) const
    {
        LengthArea eq;
        if (stressFreeState == StressFreeState::Active)
        {
            eq.l = mesh.initialLengths[i0] * lscale;
            eq.a = mesh.initialAreas  [i0] * lscale;
        }
        else
        {
            eq.l = this->length0;
            eq.a = this->area0;
        }
        return eq;
    }

    __D__ inline real3 operator()(real3 v1, real3 v2, real3 v3, EquilibriumTriangleDesc eq) const
    {
        return areaForce(v1, v2, v3, eq.a) + bondForce(v1, v2, eq.l);
    }
        
private:

    __D__ inline real3 bondForce(real3 v1, real3 v2, real l0) const
    {
        real r = max(length(v2 - v1), 1e-5_r);
        real lmax     = l0 / x0;
        real inv_lmax = x0 / l0;

        auto wlc = [this, inv_lmax] (real x) {
            return ks * inv_lmax * (4.0_r*x*x - 9.0_r*x + 6.0_r) / ( 4.0f*sqr(1.0_r - x) );
        };

        real IbforceI_wlc = wlc( min(lmax - 1e-6_r, r) * inv_lmax );

        real kp = wlc( l0 * inv_lmax ) * fastPower(l0, mpow+1);

        real IbforceI_pow = -kp / (fastPower(r, mpow+1));

        real IfI = min(forceCap, max(-forceCap, IbforceI_wlc + IbforceI_pow));

        return IfI * (v2 - v1);
    }

    __D__ inline real3 areaForce(real3 v1, real3 v2, real3 v3, real area0) const
    {
        real3 x21 = v2 - v1;
        real3 x32 = v3 - v2;
        real3 x31 = v3 - v1;

        real3 normalArea2 = cross(x21, x31);

        real area = 0.5_r * length(normalArea2);

        real coef = kd * (area - area0) / (area * area0);

        return -0.25_r * coef * cross(normalArea2, x32);
    }


    static constexpr real forceCap = 1500.0_r;
    real x0, ks, mpow;
    real kd;

    real length0, area0; ///< only useful when StressFree is false
    real lscale;
};
