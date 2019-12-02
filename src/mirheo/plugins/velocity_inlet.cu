#include "velocity_inlet.h"

#include <mirheo/core/marching_cubes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

enum {
    MAX_NEW_PARTICLE_PER_TRIANGLE = 5
};

namespace velocityInletKernels
{

__global__ void initCumulativeFluxes(real seed, int n, real *cumulativeFluxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    cumulativeFluxes[i] = Saru::uniform01(seed, i, 42 + i*i);
}

__global__ void initLocalFluxes(int n, const real3 *vertices, const real3 *velocities, real *localFluxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    real3 r0, r1, r2, v0, v1, v2;
    r0 = vertices[3*i + 0];
    r1 = vertices[3*i + 1];
    r2 = vertices[3*i + 2];

    v0 = velocities[3*i + 0];
    v1 = velocities[3*i + 1];
    v2 = velocities[3*i + 2];

    real3 nA = 0.5_r * cross(r1 - r0, r2 - r0); // normal times area of triangle
    real3 v = 0.33333_r * (v0 + v1 + v2);

    localFluxes[i] = math::abs(dot(nA, v));
}

__global__ void countFromCumulativeFluxes(int n, real dt, real numberDensity, const real *localFluxes,
                                          real *cumulativeFluxes, int *nNewParticles, int *workQueue)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nLocal = 0;
    
    if (i < n) {
        cumulativeFluxes[i] += dt * numberDensity * localFluxes[i];

        if (cumulativeFluxes[i] > 1._r) {
            nLocal = (int) cumulativeFluxes[i];
            cumulativeFluxes[i] -= nLocal;
        }
    }

    assert(nLocal >= 0);
    assert(nLocal < MAX_NEW_PARTICLE_PER_TRIANGLE);

    int warpLocalStart = warpExclusiveScan(nLocal);
    int warpStart = 0;
    const int lastWarpId = warpSize - 1;
        
    if (getLaneId<1>() == lastWarpId)
        warpStart = atomicAdd(nNewParticles, warpLocalStart + nLocal);

    warpStart = warpShfl(warpStart, lastWarpId);
    
    for (int j = 0; j < nLocal; ++j) {
        int dstId = warpStart + warpLocalStart + j;
        workQueue[dstId] = i;
    }        
}

// https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
__device__ inline real3 getBarycentricUniformTriangle(real seed, int seed0, int seed1)
{
    real r1 = Saru::uniform01(seed, 42*seed0, 5+seed1);
    real r2 = Saru::uniform01(seed, 24*seed1, 6+seed0);
    r1 = math::sqrt(r1);    
    return { (1-r1), (1-r2)*r1, r2*r1 };
}

__device__ inline real3 interpolateFrombarycentric(real3 coords, const real3 *data, int i)
{
    real3 a = data[3*i+0];
    real3 b = data[3*i+1];
    real3 c = data[3*i+2];
    return coords.x * a + coords.y * b + coords.z * c;
}

__device__ inline real3 gaussian3D(real seed, int seed0, int seed1)
{
    real2 rand1 = Saru::normal2(seed, 42*seed0, 5+seed1);
    real2 rand2 = Saru::normal2(seed, 24*seed0, 6+seed1);

    return {rand1.x, rand1.y, rand2.x}; 
}

__global__ void generateParticles(real seed, real kBT, int nNewParticles, int oldSize, PVview view, const int *workQueue,
                                  const real3 *triangles, const real3 *velocities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nNewParticles) return;
    
    int triangleId = workQueue[i];
    real3 barCoords = getBarycentricUniformTriangle(seed, i, triangleId);

    Particle p;

    p.r = interpolateFrombarycentric(barCoords, triangles,  triangleId);
    p.u = interpolateFrombarycentric(barCoords, velocities, triangleId);

    p.u += math::sqrt(kBT * view.invMass) * gaussian3D(seed, i, triangleId);
    // TODO id
    
    int dstId = oldSize + i;

    view.writeParticle(dstId, p);
}

} // namespace velocityInletKernels


VelocityInletPlugin::VelocityInletPlugin(const MirState *state, std::string name, std::string pvName,
                                         ImplicitSurfaceFunc implicitSurface, VelocityFieldFunc velocityField,
                                         real3 resolution, real numberDensity, real kBT) :
    SimulationPlugin(state, name),
    pvName(pvName),
    implicitSurface(implicitSurface),
    velocityField(velocityField),
    resolution(resolution),
    numberDensity(numberDensity),
    kBT(kBT)
{}

VelocityInletPlugin::~VelocityInletPlugin() = default;

void VelocityInletPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    std::vector<MarchingCubes::Triangle> triangles;
    MarchingCubes::computeTriangles(getState()->domain, resolution, implicitSurface, triangles);

    const int nTriangles = triangles.size();
    
    surfaceTriangles.resize_anew(nTriangles * 3);
    surfaceVelocity .resize_anew(nTriangles * 3);

    {
        size_t i = 0;
        for (const auto& t : triangles)
        {
            surfaceTriangles[i++] = t.a;
            surfaceTriangles[i++] = t.b;
            surfaceTriangles[i++] = t.c;
        }
    }

    for (size_t i = 0; i < surfaceTriangles.size(); ++i)
    {
        const real3 r = getState()->domain.local2global(surfaceTriangles[i]);
        surfaceVelocity[i] = velocityField(r);
    }

    surfaceTriangles.uploadToDevice(defaultStream);
    surfaceVelocity .uploadToDevice(defaultStream);

    cumulativeFluxes.resize_anew(nTriangles);
    localFluxes     .resize_anew(nTriangles);
    workQueue       .resize_anew(nTriangles * MAX_NEW_PARTICLE_PER_TRIANGLE);

    real seed = dist(gen);
    const int nthreads = 128;
    
    SAFE_KERNEL_LAUNCH(
        velocityInletKernels::initCumulativeFluxes,
        getNblocks(nTriangles, nthreads), nthreads, 0, defaultStream,
        seed, nTriangles, cumulativeFluxes.devPtr() );

    SAFE_KERNEL_LAUNCH(
        velocityInletKernels::initLocalFluxes,
        getNblocks(nTriangles, nthreads), nthreads, 0, defaultStream,
        nTriangles,
        surfaceTriangles.devPtr(),
        surfaceVelocity.devPtr(),
        localFluxes.devPtr() );
}

void VelocityInletPlugin::beforeCellLists(cudaStream_t stream)
{
    PVview view(pv, pv->local());
    int nTriangles = surfaceTriangles.size() / 3;
    
    workQueue.clear(stream);
    nNewParticles.clear(stream);

    const int nthreads = 128;
    
    SAFE_KERNEL_LAUNCH(
        velocityInletKernels::countFromCumulativeFluxes,
        getNblocks(nTriangles, nthreads), nthreads, 0, stream,
        nTriangles, getState()->dt, numberDensity, localFluxes.devPtr(),
        cumulativeFluxes.devPtr(), nNewParticles.devPtr(), workQueue.devPtr());

        
    nNewParticles.downloadFromDevice(stream, ContainersSynch::Synch);

    int oldSize = view.size;
    int newSize = oldSize + nNewParticles[0];

    pv->local()->resize(newSize, stream);

    view = PVview(pv, pv->local());

    real seed = dist(gen);
    
    SAFE_KERNEL_LAUNCH(
        velocityInletKernels::generateParticles,
        getNblocks(nNewParticles[0], nthreads), nthreads, 0, stream,
        seed, kBT, nNewParticles[0], oldSize, view,
        workQueue.devPtr(),
        surfaceTriangles.devPtr(),
        surfaceVelocity.devPtr() );

}

} // namespace mirheo
