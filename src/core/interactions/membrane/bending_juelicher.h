#pragma once

#include  "common.h"

__global__ void areaPerVertex(OVviewWithAreaVolume view, MeshView mesh)
{
    const int objId = blockIdx.x;
    float A = 0.0f;

    for (int i = threadIdx.x; i < mesh.ntriangles; i += blockDim.x)
    {
        int offset = objId * mesh.nvertices;
        int3 ids = mesh.triangles[i];

        float3 v0 = f4tof3( view.particles[ 2 * (offset + ids.x) ] );
        float3 v1 = f4tof3( view.particles[ 2 * (offset + ids.y) ] );
        float3 v2 = f4tof3( view.particles[ 2 * (offset + ids.z) ] );

        float A_3 = area(v1, v2, v3) * 0.3333333f;

        atomicAdd(&view.vertexAreas[offset + ids.x], A_3);
        atomicAdd(&view.vertexAreas[offset + ids.y], A_3);
        atomicAdd(&view.vertexAreas[offset + ids.z], A_3);
    }
}

