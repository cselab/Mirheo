#pragma once

#include  "common.h"

namespace BendingKantorKernels
{
struct GPU_BendingParams
{
    float cost0kb, sint0kb;
};
    
__device__  inline  float3 _fdihedral(float3 v1, float3 v2, float3 v3, float3 v4, GPU_BendingParams parameters, float3& f1)
{
    const float3 ksi   = cross(v1 - v2, v1 - v3);
    const float3 dzeta = cross(v3 - v4, v2 - v4);

    const float overIksiI   = rsqrtf(dot(ksi, ksi));
    const float overIdzetaI = rsqrtf(dot(dzeta, dzeta));

    const float cosTheta = dot(ksi, dzeta) * overIksiI * overIdzetaI;
    const float IsinThetaI2 = 1.0f - cosTheta*cosTheta;

    const float rawST_1 = rsqrtf(max(IsinThetaI2, 1.0e-6f));
    const float sinTheta_1 = copysignf( rawST_1, dot(ksi - dzeta, v4 - v1) ); // because the normals look inside
    const float beta = parameters.cost0kb - cosTheta * parameters.sint0kb * sinTheta_1;

    float b11 = -beta * cosTheta *  overIksiI   * overIksiI;
    float b12 =  beta *             overIksiI   * overIdzetaI;
    float b22 = -beta * cosTheta *  overIdzetaI * overIdzetaI;

    f1 = cross(ksi, v3 - v2)*b11 + cross(dzeta, v3 - v2)*b12;
        
    return cross(ksi, v1 - v3)*b11 + ( cross(ksi, v3 - v4) + cross(dzeta, v1 - v3) )*b12 + cross(dzeta, v3 - v4)*b22;
}

__device__ void dihedralForce(float3 v0, int locId, int rbcId,
                              const OVview& view,
                              const MembraneMeshView& mesh,
                              const GPU_BendingParams& parameters)
{
    const int offset = rbcId * mesh.nvertices;

    const int startId = mesh.maxDegree * locId;
    const int degree = mesh.degrees[locId];

    int idv1 = mesh.adjacent[startId];
    int idv2 = mesh.adjacent[startId+1];

    float3 v1 = fetchVertex(view, offset + idv1);
    float3 v2 = fetchVertex(view, offset + idv2);

    float3 f = make_float3(0.0f);

    //       v3
    //     /   \
    //   v2 --> v0
    //     \   /
    //       V
    //       v1

    float3 f0 = make_float3(0,0,0);

#pragma unroll 2
    for (int i = 0; i < degree; i++)
    {
        int idv3 = mesh.adjacent       [startId + (i+2) % degree];

        float3 v3, f1;
        v3 = fetchVertex(view, offset + idv3);

        f0 += _fdihedral(v1, v0, v2, v3, parameters, f1);

        atomicAdd(view.forces + offset + idv1, f1);
            
        v1 = v2;
        v2 = v3;

        idv1 = idv2;
        idv2 = idv3;
    }
    atomicAdd(view.forces + offset + locId, f0);
}
    
__global__ void computeBendingForces(OVview view,
                                     MembraneMeshView mesh,
                                     GPU_BendingParams parameters)
{
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    const int locId = pid % mesh.nvertices;
    const int rbcId = pid / mesh.nvertices;

    if (pid >= view.nObjects * mesh.nvertices) return;

    float3 r0 = fetchVertex(view, pid);

    dihedralForce(r0, locId, rbcId, view, mesh, parameters);
}
} // namespace BendingKantorKernels
