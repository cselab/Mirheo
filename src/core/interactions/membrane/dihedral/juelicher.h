#pragma once

#include "../fetchers.h"

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

#include <cmath>

class DihedralJuelicher : public VertexFetcherWithMeanCurvatures
{
public:    
    
    DihedralJuelicher(float kb, float C0, float kad, float DA0) :
        kb(kb),
        H0(C0/2),
        kad_pi(kad * M_PI),
        DA0(DA0),
        scurv(0)
    {}

    __D__ inline void computeCommon(const ViewType& view, int rbcId)
    {
        scurv = getScurv(view, rbcId);
    }
    
    __D__ inline float3 operator()(VertexType v0, VertexType v1, VertexType v2, VertexType v3, float3 &f1) const
    {
        float3 f0;
        float theta = supplementaryDihedralAngle(v0.r, v1.r, v2.r, v3.r);
        
        f0  = force_len   (theta, v0,     v2        );
        f0 += force_theta (       v0, v1, v2, v3, f1);
        f0 += force_area  (       v0, v1, v2        );

        return f0;
    }

private:

    __D__ inline float3 force_len(float theta, VertexType v0, VertexType v2) const
    {
        float3 d = normalize(v0.r - v2.r);
        return ( kb * (v0.H + v2.H - 2 * H0) + kad_pi * scurv ) * theta * d;
    }

    __D__ inline float3 force_theta(VertexType v0, VertexType v1, VertexType v2, VertexType v3, float3 &f1) const
    {
        float3 n, k, v20, v21, v23;

        v20 = v0.r - v2.r;
        v21 = v1.r - v2.r;
        v23 = v3.r - v2.r;
    
        n = cross(v21, v20);
        k = cross(v20, v23);

        float inv_lenn = rsqrt(dot(n,n));
        float inv_lenk = rsqrt(dot(k,k));

        float cotangent2n = dot(v20, v21) * inv_lenn;
        float cotangent2k = dot(v23, v20) * inv_lenk;
    
        float3 d1 = (dot(v20, v20)  * inv_lenn*inv_lenn) * n;
        float3 d0 =
            (-cotangent2n * inv_lenn) * n +
            (-cotangent2k * inv_lenk) * k;

        float coef = kb * (v0.H + v2.H - 2*H0)  +  kad_pi * scurv;

        f1 = coef * d1;
        return coef * d0;
    }

    __D__ inline float3 force_area(VertexType v0, VertexType v1, VertexType v2) const
    {
        float coef = -0.6666667f * kb *
            (v0.H * v0.H + v1.H * v1.H + v2.H * v2.H - 3 * H0 * H0)
            - 0.5f * kad_pi * scurv * scurv;

        float3 n  = normalize(cross(v1.r-v0.r, v2.r-v0.r));
        float3 d0 = 0.5f * cross(n, v2.r - v1.r);

        return coef * d0;
    }

    __D__ inline float getScurv(const ViewType& view, int rbcId) const
    {
        float totArea     = view.area_volumes[rbcId].x;
        float totLenTheta = view.lenThetaTot [rbcId];

        return (0.5f * totLenTheta - DA0) / totArea;
    }

    
private:    
    
    float kb, H0, kad_pi, DA0;
    float scurv;
};
