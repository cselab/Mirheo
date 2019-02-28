#pragma once

#include "../fetchers.h"
#include "../parameters.h"

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

#include <cmath>

class DihedralKantor : public VertexFetcher
{
public:    

    using ParametersType = KantorBendingParameters;
    
    DihedralKantor(ParametersType p, real lscale)        
    {
        real theta0 = p.theta / 180.0 * M_PI;
        
        cost0kb = cos(theta0) * p.kb * lscale * lscale;
        sint0kb = sin(theta0) * p.kb * lscale * lscale;
    }

    __D__ inline void computeCommon(const ViewType& view, int rbcId)
    {}
    
    __D__ inline real3 operator()(VertexType v0, VertexType v1, VertexType v2, VertexType v3, real3 &f1) const
    {
        return kantor(v1, v0, v2, v3, f1);
    }
    
private:

    __D__ inline real3 kantor(VertexType v1, VertexType v2, VertexType v3, VertexType v4, real3 &f1) const
    {
        real3 ksi   = cross(v1 - v2, v1 - v3);
        real3 dzeta = cross(v3 - v4, v2 - v4);

        real overIksiI   = rsqrtf(dot(ksi, ksi));
        real overIdzetaI = rsqrtf(dot(dzeta, dzeta));

        real cosTheta = dot(ksi, dzeta) * overIksiI * overIdzetaI;
        real IsinThetaI2 = 1.0f - cosTheta*cosTheta;

        real rawST_1 = rsqrtf(max(IsinThetaI2, 1.0e-6f));
        real sinTheta_1 = copysignf( rawST_1, dot(ksi - dzeta, v4 - v1) ); // because the normals look inside
        real beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

        real b11 = -beta * cosTheta *  overIksiI   * overIksiI;
        real b12 =  beta *             overIksiI   * overIdzetaI;
        real b22 = -beta * cosTheta *  overIdzetaI * overIdzetaI;

        f1 = cross(ksi, v3 - v2)*b11 + cross(dzeta, v3 - v2)*b12;
        
        return cross(ksi, v1 - v3)*b11 + ( cross(ksi, v3 - v4) + cross(dzeta, v1 - v3) )*b12 + cross(dzeta, v3 - v4)*b22;
    }

    real cost0kb, sint0kb;
};
