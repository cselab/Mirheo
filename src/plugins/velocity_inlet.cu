#include "velocity_inlet.h"

#include <core/marching_cubes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/kernel_launch.h>

enum {
    MAX_NEW_PARTICLE_PER_TRIANGLE = 5
};

namespace velocityInletKernels
{

__global__ void initCumulativeFluxes(float seed, int n, float *cumulativeFluxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    cumulativeFluxes[i] = Saru::uniform01(seed, i, 42 + i*i);
}

__global__ void initLocalFluxes(int n, const float3 *vertices, const float3 *velocities, float *localFluxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3 r0, r1, r2, v0, v1, v2;
    r0 = vertices[3*i + 0];
    r1 = vertices[3*i + 1];
    r2 = vertices[3*i + 2];

    v0 = velocities[3*i + 0];
    v1 = velocities[3*i + 1];
    v2 = velocities[3*i + 2];

    float3 nA = 0.5 * cross(r1 - r0, r2 - r0); // normal times area of triangle
    float3 v = 0.33333f * (v0 + v1 + v2);

    localFluxes[i] = fabs(dot(nA, v));
}

__global__ void countFromCumulativeFluxes(int n, float dt, float numberDensity, const float *localFluxes,
                                          float *cumulativeFluxes, int *nNewParticles, int *workQueue)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nLocal = 0;
    
    if (i < n) {
        cumulativeFluxes[i] += dt * numberDensity * localFluxes[i];

        if (cumulativeFluxes[i] > 1.f) {
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
__device__ inline float3 getBarycentricUniformTriangle(float seed, int seed0, int seed1)
{
    float r1 = Saru::uniform01(seed, 42*seed0, 5+seed1);
    float r2 = Saru::uniform01(seed, 24*seed1, 6+seed0);
    r1 = sqrtf(r1);    
    return { (1-r1), (1-r2)*r1, r2*r1 };
}

__device__ inline float3 interpolateFrombarycentric(float3 coords, const float3 *data, int i)
{
    float3 a = data[3*i+0];
    float3 b = data[3*i+1];
    float3 c = data[3*i+2];
    return coords.x * a + coords.y * b + coords.z * c;
}

__device__ inline float3 gaussian3D(float seed, int seed0, int seed1)
{
    float2 rand1 = Saru::normal2(seed, 42*seed0, 5+seed1);
    float2 rand2 = Saru::normal2(seed, 24*seed0, 6+seed1);

    return {rand1.x, rand1.y, rand2.x}; 
}

__global__ void generateParticles(float seed, float kBT, int nNewParticles, int oldSize, PVview view, const int *workQueue,
                                  const float3 *triangles, const float3 *velocities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nNewParticles) return;
    
    int triangleId = workQueue[i];
    float3 barCoords = getBarycentricUniformTriangle(seed, i, triangleId);

    Float3_int r, u;

    r.v = interpolateFrombarycentric(barCoords, triangles,  triangleId);
    u.v = interpolateFrombarycentric(barCoords, velocities, triangleId);

    u.v += sqrtf(kBT * view.invMass) * gaussian3D(seed, i, triangleId);

    int dstId = oldSize + i;
    
    view.particles[2*dstId+0] = r.toFloat4();
    view.particles[2*dstId+1] = u.toFloat4();
}

} // namespace velocityInletKernels


VelocityInletPlugin::VelocityInletPlugin(const YmrState *state, std::string name, std::string pvName,
                                         ImplicitSurfaceFunc implicitSurface, VelocityFieldFunc velocityField,
                                         float3 resolution, float numberDensity, float kBT) :
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
    MarchingCubes::computeTriangles(state->domain, resolution, implicitSurface, triangles);

    int nTriangles = triangles.size();
    
    surfaceTriangles.resize_anew(nTriangles * 3);
    surfaceVelocity .resize_anew(nTriangles * 3);

    int i = 0;
    for (const auto& t : triangles) {
        surfaceTriangles[i++] = t.a;
        surfaceTriangles[i++] = t.b;
        surfaceTriangles[i++] = t.c;
    }

    for (i = 0; i < surfaceTriangles.size(); ++i) {
        float3 r = state->domain.local2global(surfaceTriangles[i]);
        surfaceVelocity[i] = velocityField(r);
    }

    surfaceTriangles.uploadToDevice(defaultStream);
    surfaceVelocity .uploadToDevice(defaultStream);

    cumulativeFluxes.resize_anew(nTriangles);
    localFluxes     .resize_anew(nTriangles);
    workQueue       .resize_anew(nTriangles * MAX_NEW_PARTICLE_PER_TRIANGLE);

    float seed = dist(gen);
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
        nTriangles, state->dt, numberDensity, localFluxes.devPtr(),
        cumulativeFluxes.devPtr(), nNewParticles.devPtr(), workQueue.devPtr());

        
    nNewParticles.downloadFromDevice(stream, ContainersSynch::Synch);

    int oldSize = view.size;
    int newSize = oldSize + nNewParticles[0];

    pv->local()->resize(newSize, stream);

    view = PVview(pv, pv->local());

    float seed = dist(gen);
    
    SAFE_KERNEL_LAUNCH(
        velocityInletKernels::generateParticles,
        getNblocks(nNewParticles[0], nthreads), nthreads, 0, stream,
        seed, kBT, nNewParticles[0], oldSize, view,
        workQueue.devPtr(),
        surfaceTriangles.devPtr(),
        surfaceVelocity.devPtr() );

}

