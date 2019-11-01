#pragma once

#include <core/celllist.h>
#include <core/pvs/views/rsov.h>
#include <core/rigid/utils.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/quaternion.h>
#include <core/utils/root_finder.h>

namespace ShapeBounceKernels
{

template <class Shape>
__device__ static inline real3 rescue(real3 candidate, real dt, real tol, const Shape& shape)
{
    const int maxIters = 100;
    const real factor = 50.0_r * dt;

    for (int i = 0; i < maxIters; i++)
    {
        const real v = shape.inOutFunction(candidate);
        if (v > tol) break;

        const real seed0 = candidate.x - math::floor(candidate.x) / i;
        const real seed1 = candidate.y - math::floor(candidate.y) * i;
        const real seed2 = candidate.z - math::floor(candidate.z) + i;
        
        real3 rndShift;
        rndShift.x = Saru::mean0var1(seed0, seed1, seed2);
        rndShift.y = Saru::mean0var1(rndShift.x, seed0, seed1);
        rndShift.z = Saru::mean0var1(rndShift.y, seed2, seed1 * seed0);

        if (shape.inOutFunction(candidate + factor * rndShift) > v)
            candidate += factor * rndShift;
    }

    return candidate;
}

template <class Shape, class BounceKernel>
__device__ static inline void bounceCellArray(
        const RSOVviewWithOldMotion<Shape>& ovView, PVviewWithOldParticles& pvView,
        int objId, const int *validCells, int nCells,
        CellListInfo cinfo, const real dt, const BounceKernel& bounceKernel)
{
    const real threshold = 2e-5_r;

    const Shape& shape = ovView.shape;
    
    const auto motion     = toRealMotion( ovView.motions[objId] );
    const auto old_motion = toRealMotion( ovView.old_motions[objId] );

    if (threadIdx.x >= nCells) return;

    const int cid = validCells[threadIdx.x];
    const int pstart = cinfo.cellStarts[cid];
    const int pend   = cinfo.cellStarts[cid+1];

    // XXX: changing reading layout may improve performance here
    for (int pid = pstart; pid < pend; pid++)
    {
        Particle p (pvView.readParticle   (pid));
        const auto rOld = pvView.readOldPosition(pid);

        // Go to the obj frame of reference
        const real3 coo    = Quaternion::rotate(p.r     - motion.r,  Quaternion::conjugate(    motion.q));
        const real3 oldCoo = Quaternion::rotate(rOld - old_motion.r, Quaternion::conjugate(old_motion.q));
        const real3 dr = coo - oldCoo;

        // If the particle is outside - skip it, it's fine
        if (shape.inOutFunction(coo) > 0.0_r) continue;

        real3 newCoo;
        
        // worst case scenario: was already inside before, we need to rescue the particle
        if (shape.inOutFunction(oldCoo) <= 0.0_r)
        {
            newCoo = rescue(coo, dt, threshold, shape);

            if (shape.inOutFunction(newCoo) < 0.0_r)
            {
                printf("Bounce-back rescue failed on particle %ld (%g %g %g) (local: (%g %g %g))  %g -> %g (rescued: %g).\n",
                       p.getId(),
                       p.r.x, p.r.y, p.r.z,
                       coo.x, coo.y, coo.z,
                       shape.inOutFunction(oldCoo),
                       shape.inOutFunction(coo),
                       shape.inOutFunction(newCoo));

                newCoo = oldCoo;
            }
        }
        // otherwise find intersection and perform the bounce
        else
        {
            // This is intersection point
            constexpr RootFinder::Bounds limits {0._r, 1._r};
            const real alpha = RootFinder::linearSearch( [=] (const real lambda) { return shape.inOutFunction(oldCoo + dr*lambda);}, limits );
            newCoo = oldCoo + dr*max(alpha, limits.lo);

            // Push out a little bit
            const real3 normal = shape.normal(newCoo);
            newCoo += threshold * normal;

            // If smth went notoriously bad
            if (shape.inOutFunction(newCoo) < 0.0_r)
            {
                printf("Bounce-back failed on particle %ld (%g %g %g) (local: (%g %g %g))  %g -> %g to %g, alpha %f. Recovering to old position\n",
                       p.getId(),
                       p.r.x, p.r.y, p.r.z,
                       coo.x, coo.y, coo.z,
                       shape.inOutFunction(oldCoo),
                       shape.inOutFunction(coo),
                       shape.inOutFunction(newCoo - threshold*normal),
                       alpha);

                newCoo = oldCoo;
            }
        }

        real3 normal = shape.normal(newCoo);
        
        // Return to the original frame
        newCoo = Quaternion::rotate(newCoo, motion.q) + motion.r;
        normal = Quaternion::rotate(normal, motion.q);

        // Change velocity's frame to the object frame, correct for rotation as well
        const real3 vObj = motion.vel + cross( motion.omega, newCoo-motion.r );
        const real3 newU = bounceKernel.newVelocity(p.u, vObj, normal, pvView.mass);

        const real3 frc = -pvView.mass * (newU - p.u) / dt;
        atomicAdd( &ovView.motions[objId].force,  make_rigidReal3(frc));
        atomicAdd( &ovView.motions[objId].torque, make_rigidReal3(cross(newCoo - motion.r, frc)) );

        p.r = newCoo;
        p.u = newU;
        pvView.writeParticle(pid, p);
    }
}

template <class Shape>
__device__ static inline bool isValidCell(int3 cid3, const RealRigidMotion& motion, CellListInfo cinfo, const Shape& shape)
{
    constexpr real threshold = 0.5_r;

    real3 v000 = make_real3(cid3) * cinfo.h - cinfo.localDomainSize * 0.5_r - motion.r;
    const real4 invq = Quaternion::conjugate(motion.q);

    const real3 v001 = Quaternion::rotate( v000 + make_real3(        0,         0, cinfo.h.z), invq );
    const real3 v010 = Quaternion::rotate( v000 + make_real3(        0, cinfo.h.y,         0), invq );
    const real3 v011 = Quaternion::rotate( v000 + make_real3(        0, cinfo.h.y, cinfo.h.z), invq );
    const real3 v100 = Quaternion::rotate( v000 + make_real3(cinfo.h.x,         0,         0), invq );
    const real3 v101 = Quaternion::rotate( v000 + make_real3(cinfo.h.x,         0, cinfo.h.z), invq );
    const real3 v110 = Quaternion::rotate( v000 + make_real3(cinfo.h.x, cinfo.h.y,         0), invq );
    const real3 v111 = Quaternion::rotate( v000 + make_real3(cinfo.h.x, cinfo.h.y, cinfo.h.z), invq );

    v000 = Quaternion::rotate( v000, invq );

    return ( shape.inOutFunction(v000) < threshold ||
             shape.inOutFunction(v001) < threshold ||
             shape.inOutFunction(v010) < threshold ||
             shape.inOutFunction(v011) < threshold ||
             shape.inOutFunction(v100) < threshold ||
             shape.inOutFunction(v101) < threshold ||
             shape.inOutFunction(v110) < threshold ||
             shape.inOutFunction(v111) < threshold );
}

template <class Shape, class BounceKernel>
__global__ void bounce(RSOVviewWithOldMotion<Shape> ovView, PVviewWithOldParticles pvView,
                       CellListInfo cinfo, const real dt, const BounceKernel bounceKernel)
{
    // About max travel distance per step + safety
    // Safety comes from the fact that bounce works with the analytical shape,
    //  and extent is computed w.r.t. particles
    const real tol = 1.5_r;

    const int objId = blockIdx.x;
    const int tid = threadIdx.x;
    if (objId >= ovView.nObjects) return;

    // Preparation step. Filter out all the cells that don't intersect the surface
    __shared__ int nCells;
    extern __shared__ int validCells[];

    if (tid == 0)
        nCells = 0;
    __syncthreads();

    const int3 cidLow  = cinfo.getCellIdAlongAxes(ovView.comAndExtents[objId].low  - tol);
    const int3 cidHigh = cinfo.getCellIdAlongAxes(ovView.comAndExtents[objId].high + tol);

    const int3 span = cidHigh - cidLow + make_int3(1,1,1);
    const int totCells = span.x * span.y * span.z;

    for (int i = tid; i-tid < totCells; i += blockDim.x)
    {
        const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
        const int cid = cinfo.encode(cid3);

        if ( i < totCells &&
             cid < cinfo.totcells &&
             isValidCell(cid3, toRealMotion(ovView.motions[objId]), cinfo, ovView.shape) )
        {
            const int id = atomicAggInc(static_cast<int*>(&nCells));
            validCells[id] = cid;
        }

        __syncthreads();

        // If we have enough cells ready - process them
        if (nCells >= blockDim.x)
        {
            bounceCellArray(ovView, pvView, objId, validCells, blockDim.x, cinfo, dt, bounceKernel);

            __syncthreads();

            if (tid == 0) nCells -= blockDim.x;
            validCells[tid] = validCells[tid + blockDim.x];

            __syncthreads();
        }
    }

    __syncthreads();

    // Process remaining
    bounceCellArray(ovView, pvView, objId, validCells, nCells, cinfo, dt, bounceKernel);
}


} // namespace EllipsoidBounceKernels
