#pragma once

#include "../fetchers.h"

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

#include <cmath>

class DihedralKantor : public VertexFetcher
{
public:    
    
    DihedralKantor(float kb, float theta0) :
        cost0kb(cos(theta0 / 180.0 * M_PI) * kb),
        sint0kb(sin(theta0 / 180.0 * M_PI) * kb)
    {}

    __D__ inline void computeCommon(const ViewType& view, int rbcId)
    {}
    
    __D__ inline float3 operator()(VertexType v1, VertexType v2, VertexType v3, VertexType v4, float3 &f1) const
    {
        float3 ksi   = cross(v1 - v2, v1 - v3);
        float3 dzeta = cross(v3 - v4, v2 - v4);

        float overIksiI   = rsqrtf(dot(ksi, ksi));
        float overIdzetaI = rsqrtf(dot(dzeta, dzeta));

        float cosTheta = dot(ksi, dzeta) * overIksiI * overIdzetaI;
        float IsinThetaI2 = 1.0f - cosTheta*cosTheta;

        float rawST_1 = rsqrtf(max(IsinThetaI2, 1.0e-6f));
        float sinTheta_1 = copysignf( rawST_1, dot(ksi - dzeta, v4 - v1) ); // because the normals look inside
        float beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

        float b11 = -beta * cosTheta *  overIksiI   * overIksiI;
        float b12 =  beta *             overIksiI   * overIdzetaI;
        float b22 = -beta * cosTheta *  overIdzetaI * overIdzetaI;

        f1 = cross(ksi, v3 - v2)*b11 + cross(dzeta, v3 - v2)*b12;
        
        return cross(ksi, v1 - v3)*b11 + ( cross(ksi, v3 - v4) + cross(dzeta, v1 - v3) )*b12 + cross(dzeta, v3 - v4)*b22;
    }
    
private:

    float cost0kb, sint0kb;
};
