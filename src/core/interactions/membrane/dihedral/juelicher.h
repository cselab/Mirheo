#pragma once

#include "../fetchers.h"
#include "../parameters.h"

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

#include <cmath>

class DihedralJuelicher : public VertexFetcherWithMeanCurvatures
{
public:    

    using ParametersType = JuelicherBendingParameters;
    
    DihedralJuelicher(ParametersType p, real lscale) :
        scurv(0)
    {
        kb     = p.kb         * lscale*lscale;
        kad_pi = p.kad * M_PI * lscale*lscale;

        H0  = p.C0 / (2*lscale);        
        DA0 = p.DA0 / (lscale*lscale);
    }

    __D__ inline void computeCommon(const ViewType& view, int rbcId)
    {
        scurv = getScurv(view, rbcId);
    }
    
    __D__ inline real3 operator()(VertexType v0, VertexType v1, VertexType v2, VertexType v3, real3 &f1) const
    {
        real3 f0;
        real theta = supplementaryDihedralAngle(v0.r, v1.r, v2.r, v3.r);
        
        f0  = force_len   (theta, v0,     v2        );
        f0 += force_theta (       v0, v1, v2, v3, f1);
        f0 += force_area  (       v0, v1, v2        );

        return f0;
    }

private:

    __D__ inline real3 force_len(real theta, VertexType v0, VertexType v2) const
    {
        real3 d = normalize(v0.r - v2.r);
        return ( kb * (v0.H + v2.H - 2 * H0) + kad_pi * scurv ) * theta * d;
    }

    __D__ inline real3 force_theta(VertexType v0, VertexType v1, VertexType v2, VertexType v3, real3 &f1) const
    {
        real3 n, k, v20, v21, v23;

        v20 = v0.r - v2.r;
        v21 = v1.r - v2.r;
        v23 = v3.r - v2.r;
    
        n = cross(v21, v20);
        k = cross(v20, v23);

        real inv_lenn = rsqrt(dot(n,n));
        real inv_lenk = rsqrt(dot(k,k));

        real cotangent2n = dot(v20, v21) * inv_lenn;
        real cotangent2k = dot(v23, v20) * inv_lenk;
    
        real3 d1 = (dot(v20, v20)  * inv_lenn*inv_lenn) * n;
        real3 d0 =
            (-cotangent2n * inv_lenn) * n +
            (-cotangent2k * inv_lenk) * k;

        real coef = kb * (v0.H + v2.H - 2*H0)  +  kad_pi * scurv;

        f1 = coef * d1;
        return coef * d0;
    }

    __D__ inline real3 force_area(VertexType v0, VertexType v1, VertexType v2) const
    {
        real coef = -0.6666667_r * kb *
            (v0.H * v0.H + v1.H * v1.H + v2.H * v2.H - 3 * H0 * H0)
            - 0.5_r * kad_pi * scurv * scurv;

        real3 n  = normalize(cross(v1.r-v0.r, v2.r-v0.r));
        real3 d0 = 0.5_r * cross(n, v2.r - v1.r);

        return coef * d0;
    }

    __D__ inline real getScurv(const ViewType& view, int rbcId) const
    {
        real totArea     = view.area_volumes[rbcId].x;
        real totLenTheta = view.lenThetaTot [rbcId];

        return (0.5_r * totLenTheta - DA0) / totArea;
    }

    
private:    
    
    real kb, H0, kad_pi, DA0;
    real scurv;
};
