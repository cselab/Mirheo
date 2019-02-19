#include "radial_velocity_control.h"
#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <plugins/simple_serializer.h>

namespace RadialVelocityControlKernels {

__device__ inline bool validRadius(float r2, float minR2, float maxR2)
{
    return  minR2 < r2 && r2 < maxR2;
}

__global__ void addForce(PVview view, DomainInfo domain, float minRadiusSquare, float maxRadiusSquare,
                         float3 center, float forceFactor)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    Particle p;
    p.readCoordinate(view.particles, gid);
    float3 r = domain.local2global(p.r);
    r -= center;
    float r2 = r.x * r.x + r.y * r.y;

    if (!validRadius(r2, minRadiusSquare, maxRadiusSquare))
        return;

    float factor = forceFactor / r2;
    
    float3 force = {r.x * factor,
                    r.y * factor,
                    0.f};
    
    view.forces[gid] += make_float4(force, 0.0f);
}

__global__ void sumVelocity(PVview view, DomainInfo domain, float minRadiusSquare, float maxRadiusSquare,
                            float3 center, double *totVel, int *nSamples)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float ur = 0.f;

    if (gid < view.size) {
        
        Particle p(view.particles, gid);
        float3 r = domain.local2global(p.r);
        r -= center;

        float r2 = r.x * r.x + r.y * r.y;

        if (validRadius(r2, minRadiusSquare, maxRadiusSquare)) {
            atomicAggInc(nSamples);
            ur = r.x * p.u.x + r.y * p.u.y;
        }
    }
    
    double urSum = warpReduce(ur, [](float a, float b) { return a+b; });
    if (threadIdx.x % warpSize == 0)
        atomicAdd(totVel, urSum);
}

} // namespace RadialVelocityControlKernels

SimulationRadialVelocityControl::SimulationRadialVelocityControl(const YmrState *state, std::string name, std::vector<std::string> pvNames,
                                                                 float minRadius, float maxRadius, int sampleEvery, int tuneEvery, int dumpEvery,
                                                                 float3 center, float targetVel, float Kp, float Ki, float Kd) :
    SimulationPlugin(state, name),
    pvNames(pvNames),
    minRadiusSquare(minRadius * minRadius),
    maxRadiusSquare(maxRadius * maxRadius),
    currentVel(0),
    targetVel(targetVel),
    center(center),
    sampleEvery(sampleEvery),
    tuneEvery(tuneEvery),
    dumpEvery(dumpEvery), 
    force(0),
    pid(0, Kp, Ki, Kd),
    accumulatedTotVel(0)
{}

SimulationRadialVelocityControl::~SimulationRadialVelocityControl() = default;

void SimulationRadialVelocityControl::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    for (auto &pvName : pvNames)
        pvs.push_back(simulation->getPVbyNameOrDie(pvName));
}

void SimulationRadialVelocityControl::beforeForces(cudaStream_t stream)
{
    for (auto &pv : pvs)
    {
        PVview view(pv, pv->local());
        const int nthreads = 128;

        SAFE_KERNEL_LAUNCH
            (RadialVelocityControlKernels::addForce,
             getNblocks(view.size, nthreads), nthreads, 0, stream,
             view, state->domain, minRadiusSquare, maxRadiusSquare, center, force );
    }
}

void SimulationRadialVelocityControl::sampleOnePv(ParticleVector *pv, cudaStream_t stream) {
    PVview pvView(pv, pv->local());
    const int nthreads = 128;
 
    SAFE_KERNEL_LAUNCH
        (RadialVelocityControlKernels::sumVelocity,
         getNblocks(pvView.size, nthreads), nthreads, 0, stream,
         pvView, state->domain, minRadiusSquare, maxRadiusSquare, center, totVel.devPtr(), nSamples.devPtr());
}

void SimulationRadialVelocityControl::afterIntegration(cudaStream_t stream)
{
    if (state->currentStep % sampleEvery == 0 && state->currentStep != 0)
    {
        debug2("Velocity control %s is sampling now", name.c_str());

        totVel.clearDevice(stream);
        for (auto &pv : pvs)
            sampleOnePv(pv, stream);
        totVel.downloadFromDevice(stream);
        accumulatedTotVel += totVel[0];
    }
    
    if (state->currentStep % tuneEvery != 0 || state->currentStep == 0)
        return;
    
    nSamples.downloadFromDevice(stream);
    nSamples.clearDevice(stream);
    
    long nSamples_loc, nSamples_tot = 0;
    double totVel_tot = 0;  

    nSamples_loc = nSamples[0];
    
    MPI_Check( MPI_Allreduce(&nSamples_loc,        &nSamples_tot, 1, MPI_LONG,   MPI_SUM, comm) );
    MPI_Check( MPI_Allreduce(&accumulatedTotVel,   &totVel_tot,   1, MPI_DOUBLE, MPI_SUM, comm) );

    currentVel = nSamples_tot ? totVel_tot / nSamples_tot : 0.f;
    force = pid.update(targetVel - currentVel);
    accumulatedTotVel = 0;
}

void SimulationRadialVelocityControl::serializeAndSend(cudaStream_t stream)
{
    if (state->currentStep % dumpEvery != 0 || state->currentStep == 0)
        return;

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, state->currentTime, state->currentStep, currentVel, force);
    send(sendBuffer);
}




PostprocessRadialVelocityControl::PostprocessRadialVelocityControl(std::string name, std::string filename) :
    PostprocessPlugin(name)
{
    fdump = fopen(filename.c_str(), "w");
    if (!fdump) die("Could not open file '%s'", filename.c_str());
    fprintf(fdump, "# time time_step velocity*r force/r**3\n");
}

PostprocessRadialVelocityControl::~PostprocessRadialVelocityControl()
{
    fclose(fdump);
}

void PostprocessRadialVelocityControl::deserialize(MPI_Status& stat)
{
    int currentTimeStep;
    TimeType currentTime;
    float vel, force;

    SimpleSerializer::deserialize(data, currentTime, currentTimeStep, vel, force);

    if (rank == 0) {
        fprintf(fdump, "%g %d %g %g\n", currentTime, currentTimeStep, vel, force);        
        fflush(fdump);
    }
}
