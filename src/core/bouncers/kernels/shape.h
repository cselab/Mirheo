#pragma once

#include <core/bounce_solver.h>
#include <core/celllist.h>
#include <core/pvs/views/rsov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/quaternion.h>

namespace ShapeBounceKernels
{

template <class Shape>
__device__ inline void bounceCellArray(
        const RSOVviewWithOldMotion<Shape>& ovView, PVviewWithOldParticles& pvView,
        int objId,
        int *validCells, int nCells,
        CellListInfo cinfo, const float dt)
{
    const float threshold = 2e-5f;

    const Shape& shape = ovView.shape;
    
    auto motion     = toSingleMotion( ovView.motions[objId] );
    auto old_motion = toSingleMotion( ovView.old_motions[objId] );

    if (threadIdx.x >= nCells) return;

    int cid = validCells[threadIdx.x];
    int pstart = cinfo.cellStarts[cid];
    int pend   = cinfo.cellStarts[cid+1];

    // XXX: changing reading layout may improve performance here
    for (int pid = pstart; pid < pend; pid++)
    {
        Particle p (pvView.readParticle   (pid));
        auto rOld = pvView.readOldPosition(pid);

        // Go to the obj frame of reference
        float3 coo    = rotate(p.r     - motion.r,     invQ(motion.q));
        float3 oldCoo = rotate(rOld - old_motion.r, invQ(old_motion.q));
        float3 dr = coo - oldCoo;

        // If the particle is outside - skip it, it's fine
        if (shape.inOutFunction(coo) > 0.0f) continue;

        // This is intersection point
        float alpha = solveLinSearch( [=] (const float lambda) { return shape.inOutFunction(oldCoo + dr*lambda);} );
        float3 newCoo = oldCoo + dr*max(alpha, 0.0f);

        // TODO
        // Push out a little bit
        // float3 normal = normalize(make_float3(
        //         axes.y*axes.y * axes.z*axes.z * newCoo.x,
        //         axes.z*axes.z * axes.x*axes.x * newCoo.y,
        //         axes.x*axes.x * axes.y*axes.y * newCoo.z));

        float3 normal{0.f, 0.f, 0.f};
        newCoo += threshold*normal;

        // If smth went notoriously bad
        if (shape.inOutFunction(newCoo) < 0.0f)
        {
            printf("Bounce-back failed on particle %d (%f %f %f)  %f -> %f to %f, alpha %f. Recovering to old position\n",
                    p.i1, p.r.x, p.r.y, p.r.z,
                    shape.inOutFunction(oldCoo), shape.inOutFunction(coo),
                    shape.inOutFunction(newCoo - threshold*normal), alpha);

            newCoo = oldCoo;
        }

        // Return to the original frame
        newCoo = rotate(newCoo, motion.q) + motion.r;

        // Change velocity's frame to the object frame, correct for rotation as well
        float3 vEll = motion.vel + cross( motion.omega, newCoo-motion.r );
        float3 newU = vEll - (p.u - vEll);

        const float3 frc = -pvView.mass * (newU - p.u) / dt;
        atomicAdd( &ovView.motions[objId].force,  make_rigidReal3(frc));
        atomicAdd( &ovView.motions[objId].torque, make_rigidReal3(cross(newCoo - motion.r, frc)) );

        p.r = newCoo;
        p.u = newU;
        pvView.writeParticle(pid, p);
    }
}

template <class Shape>
__device__ inline bool isValidCell(int3 cid3, SingleRigidMotion motion, CellListInfo cinfo, const Shape& shape)
{
    const float threshold = 0.5f;

    float3 v000 = make_float3(cid3) * cinfo.h - cinfo.localDomainSize*0.5f - motion.r;
    const float4 invq = invQ(motion.q);

    float3 v001 = rotate( v000 + make_float3(        0,         0, cinfo.h.z), invq );
    float3 v010 = rotate( v000 + make_float3(        0, cinfo.h.y,         0), invq );
    float3 v011 = rotate( v000 + make_float3(        0, cinfo.h.y, cinfo.h.z), invq );
    float3 v100 = rotate( v000 + make_float3(cinfo.h.x,         0,         0), invq );
    float3 v101 = rotate( v000 + make_float3(cinfo.h.x,         0, cinfo.h.z), invq );
    float3 v110 = rotate( v000 + make_float3(cinfo.h.x, cinfo.h.y,         0), invq );
    float3 v111 = rotate( v000 + make_float3(cinfo.h.x, cinfo.h.y, cinfo.h.z), invq );

    v000 = rotate( v000, invq );

    return ( shape.inOutFunction(v000) < threshold ||
             shape.inOutFunction(v001) < threshold ||
             shape.inOutFunction(v010) < threshold ||
             shape.inOutFunction(v011) < threshold ||
             shape.inOutFunction(v100) < threshold ||
             shape.inOutFunction(v101) < threshold ||
             shape.inOutFunction(v110) < threshold ||
             shape.inOutFunction(v111) < threshold );
}

template <class Shape>
__global__ void bounce(RSOVviewWithOldMotion<Shape> ovView, PVviewWithOldParticles pvView,
                       CellListInfo cinfo, const float dt)
{
    // About max travel distance per step + safety
    // Safety comes from the fact that bounce works with the analytical shape,
    //  and extent is computed w.r.t. particles
    const int tol = 1.5f;

    const int objId = blockIdx.x;
    const int tid = threadIdx.x;
    if (objId >= ovView.nObjects) return;

    // Preparation step. Filter out all the cells that don't intersect the surface
    __shared__ volatile int nCells;
    extern __shared__ int validCells[];

    nCells = 0;
    __syncthreads();

    const int3 cidLow  = cinfo.getCellIdAlongAxes(ovView.comAndExtents[objId].low  - tol);
    const int3 cidHigh = cinfo.getCellIdAlongAxes(ovView.comAndExtents[objId].high + tol);

    const int3 span = cidHigh - cidLow + make_int3(1,1,1);
    const int totCells = span.x * span.y * span.z;

    for (int i=tid; i-tid < totCells; i+=blockDim.x)
    {
        const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
        const int cid = cinfo.encode(cid3);

        if ( i < totCells &&
             cid < cinfo.totcells &&
             isValidCell(cid3, toSingleMotion(ovView.motions[objId]), cinfo, ovView.shape) )
        {
            int id = atomicAggInc((int*)&nCells);
            validCells[id] = cid;
        }

        __syncthreads();

        // If we have enough cells ready - process them
        if (nCells >= blockDim.x)
        {
            bounceCellArray(ovView, pvView, objId, validCells, blockDim.x, cinfo, dt);

            __syncthreads();

            if (tid == 0) nCells -= blockDim.x;
            validCells[tid] = validCells[tid + blockDim.x];

            __syncthreads();
        }
    }

    __syncthreads();

    // Process remaining
    bounceCellArray(ovView, pvView, objId, validCells, nCells, cinfo, dt);
}


} // namespace EllipsoidBounceKernels
