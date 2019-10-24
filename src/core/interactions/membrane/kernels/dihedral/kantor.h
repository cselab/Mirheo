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
    
    DihedralKantor(ParametersType p, mReal lscale)
    {
        const mReal theta0 = p.theta / 180.0 * M_PI;
        
        cost0kb = cos(theta0) * p.kb * lscale * lscale;
        sint0kb = sin(theta0) * p.kb * lscale * lscale;
    }

    __D__ inline void computeCommon(const ViewType& view, int rbcId)
    {}
    
    __D__ inline mReal3 operator()(VertexType v0, VertexType v1, VertexType v2, VertexType v3, mReal3 &f1) const
    {
        return kantor(v1, v0, v2, v3, f1);
    }
    
private:

    __D__ inline mReal3 kantor(VertexType v1, VertexType v2, VertexType v3, VertexType v4, mReal3 &f1) const
    {
        const mReal3 ksi   = cross(v1 - v2, v1 - v3);
        const mReal3 dzeta = cross(v3 - v4, v2 - v4);

        const mReal overIksiI   = math::rsqrt(dot(ksi, ksi));
        const mReal overIdzetaI = math::rsqrt(dot(dzeta, dzeta));

        const mReal cosTheta = dot(ksi, dzeta) * overIksiI * overIdzetaI;
        const mReal IsinThetaI2 = 1.0f - cosTheta*cosTheta;

        const mReal rawST_1 = math::rsqrt(max(IsinThetaI2, 1.0e-6f));
        const mReal sinTheta_1 = copysignf( rawST_1, dot(ksi - dzeta, v4 - v1) ); // because the normals look inside
        const mReal beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

        const mReal b11 = -beta * cosTheta *  overIksiI   * overIksiI;
        const mReal b12 =  beta *             overIksiI   * overIdzetaI;
        const mReal b22 = -beta * cosTheta *  overIdzetaI * overIdzetaI;

        f1 = cross(ksi, v3 - v2)*b11 + cross(dzeta, v3 - v2)*b12;
        
        return cross(ksi, v1 - v3)*b11 + ( cross(ksi, v3 - v4) + cross(dzeta, v1 - v3) )*b12 + cross(dzeta, v3 - v4)*b22;
    }

    mReal cost0kb, sint0kb;
};
