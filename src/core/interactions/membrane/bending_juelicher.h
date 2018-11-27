#pragma once

#include  "common.h"

__device__ inline float3 fetchVertex(OVviewWithJuelicherQuants view, int i)
{
    // 2 because of float4
    return Float3_int(view.particles[2 * i]).v;
}

__device__ inline float compute_lenTheta(float3 v0, float3 v1, float3 v2, float3 v3)
{
    float len = length(v2 - v0);
    float theta = supplementaryDihedralAngle(v0, v1, v2, v3);
    return len * theta;
}

__global__ void lenThetaPerVertex(OVviewWithJuelicherQuants view, MembraneMeshView mesh)
{
    assert(view.objSize == mesh.nvertices);

    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    int locId = pid % mesh.nvertices;
    int rbcId = pid / mesh.nvertices;
    int offset = rbcId * mesh.nvertices;

    if (pid >= view.nObjects * mesh.nvertices) return;

    int startId = mesh.maxDegree * locId;
    int degree = mesh.degrees[locId];

    int idv1 = mesh.adjacent[startId];
    int idv2 = mesh.adjacent[startId+1];

    float3 v0 = fetchVertex(view, pid);
    float3 v1 = fetchVertex(view, offset + idv1);
    float3 v2 = fetchVertex(view, offset + idv2);

#pragma unroll 2
    for (int i = 0; i < degree; i++) {

        int idv3 = mesh.adjacent[startId + (i+2) % degree];
        float3 v3 = fetchVertex(view, offset + idv3);

        view.vertexAreas     [pid] += 0.3333333f * triangleArea(v0, v1, v2);
        view.vertexLenThetas [pid] += compute_lenTheta(v0, v1, v2, v3);

        v1 = v2;
        v2 = v3;
    }    
}

