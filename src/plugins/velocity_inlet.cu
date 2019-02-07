#include "velocity_inlet.h"

#include <core/marching_cubes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/kernel_launch.h>

namespace velocityInletKernels {

__global__ void initCumulativeSum(float seed, int n, float *cumulativeSums)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    cumulativeSums[i] = Saru::uniform01(seed, i, 42 + i*i);
}

}


VelocityInletPlugin::VelocityInletPlugin(const YmrState *state, std::string name, std::string pvName,
                                         ImplicitSurfaceFunc implicitSurface, VelocityFieldFunc velocityField,
                                         float3 resolution) :
    SimulationPlugin(state, name),
    pvName(pvName),
    implicitSurface(implicitSurface),
    velocityField(velocityField),
    resolution(resolution)
{}

VelocityInletPlugin::~VelocityInletPlugin() = default;

void VelocityInletPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    std::vector<MarchingCubes::Triangle> triangles;
    MarchingCubes::computeTriangles(state->domain, resolution, implicitSurface, triangles);

    surfaceTriangles.resize_anew(triangles.size() * 3);
    surfaceVelocity .resize_anew(triangles.size() * 3);

    int i = 0;
    for (const auto& t : triangles) {
        surfaceTriangles[i++] = t.a;
        surfaceTriangles[i++] = t.b;
        surfaceTriangles[i++] = t.c;
    }

    for (i = 0; i < surfaceTriangles.size(); ++i)
        surfaceVelocity[i] = velocityField(surfaceTriangles[i]);        

    surfaceTriangles.uploadToDevice(0);
    surfaceVelocity .uploadToDevice(0);

    cummulativeSum.resize_anew(triangles.size());

    float seed = 0.42424242;
    const int nthreads = 128;
    
    SAFE_KERNEL_LAUNCH(
        velocityInletKernels::initCumulativeSum,
        getNblocks(cummulativeSum.size(), nthreads), nthreads, 0, 0,
        seed, cummulativeSum.size(), cummulativeSum.devPtr() );

}

void VelocityInletPlugin::beforeParticleDistribution(cudaStream_t stream)
{
    // DomainInfo domain = state->domain;
    // PVview view(pv, pv->local());

    // const int nthreads = 128;

    // numberCrossedParticles.clear(stream);
    
    // SAFE_KERNEL_LAUNCH(
    //         exchange_pvs_flux_plane_kernels::countParticles,
    //         getNblocks(view1.size, nthreads), nthreads, 0, stream,
    //         domain, view1, plane, numberCrossedParticles.devPtr() );

    // numberCrossedParticles.downloadFromDevice(stream, ContainersSynch::Synch);

    // const int old_size2 = view2.size;
    // const int new_size2 = old_size2 + numberCrossedParticles.hostPtr()[0];

    // pv2->local()->resize(new_size2, stream);
    // numberCrossedParticles.clear(stream);

    // view2 = PVview(pv2, pv2->local());

    // SAFE_KERNEL_LAUNCH(
    //                    exchange_pvs_flux_plane_kernels::moveParticles,
    //                    getNblocks(view1.size, nthreads), nthreads, 0, stream,
    //                    domain, view1, view2, plane, old_size2, numberCrossedParticles.devPtr() );

}

