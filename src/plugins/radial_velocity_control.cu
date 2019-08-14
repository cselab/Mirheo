#include "radial_velocity_control.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/restart_helpers.h>

namespace RadialVelocityControlKernels
{

__device__ inline bool validRadius(float r2, float minR2, float maxR2)
{
    return  minR2 < r2 && r2 < maxR2;
}

__global__ void addForce(PVview view, DomainInfo domain, float minRadiusSquare, float maxRadiusSquare,
                         float3 center, float forceFactor)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    float3 r = make_float3(view.readPosition(gid));
    r = domain.local2global(r);
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
        
        Particle p(view.readParticle(gid));
        float3 r = domain.local2global(p.r);
        r -= center;

        float r2 = r.x * r.x + r.y * r.y;

        if (validRadius(r2, minRadiusSquare, maxRadiusSquare)) {
            atomicAggInc(nSamples);
            ur = r.x * p.u.x + r.y * p.u.y;
        }
    }
    
    double urSum = warpReduce(ur, [](float a, float b) { return a+b; });

    if (laneId() == 0)
        atomicAdd(totVel, urSum);
}

} // namespace RadialVelocityControlKernels

SimulationRadialVelocityControl::SimulationRadialVelocityControl(const MirState *state, std::string name, std::vector<std::string> pvNames,
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
    accumulatedVel(0),
    accumulatedSamples(0)
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
    if (isTimeEvery(state, sampleEvery))
    {
        debug2("Velocity control %s is sampling now", name.c_str());

        totVel.clearDevice(stream);
        nSamples.clearDevice(stream);

        for (auto &pv : pvs)
            sampleOnePv(pv, stream);

        totVel  .downloadFromDevice(stream, ContainersSynch::Asynch);
        nSamples.downloadFromDevice(stream, ContainersSynch::Synch);

        accumulatedVel     += totVel[0];
        accumulatedSamples += nSamples[0];
    }
    
    if (!isTimeEvery(state, tuneEvery)) return;
    
    
    unsigned long long nSamples_tot = 0;
    long double totVel_tot = 0;  
    
    MPI_Check( MPI_Allreduce(&accumulatedSamples, &nSamples_tot, 1, MPI_UNSIGNED_LONG_LONG,   MPI_SUM, comm) );
    MPI_Check( MPI_Allreduce(&accumulatedVel,     &totVel_tot,   1, MPI_LONG_DOUBLE,          MPI_SUM, comm) );

    currentVel = nSamples_tot ? totVel_tot / nSamples_tot : 0.f;
    force = pid.update(targetVel - currentVel);

    accumulatedVel     = 0;
    accumulatedSamples = 0;
}

void SimulationRadialVelocityControl::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery))
        return;

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, state->currentTime, state->currentStep, currentVel, force);
    send(sendBuffer);
}

void SimulationRadialVelocityControl::checkpoint(MPI_Comm comm, std::string path, int checkpointId)
{
    auto filename = createCheckpointNameWithId(path, "plugin." + name, "txt", checkpointId);

    TextIO::write(filename, pid);
    
    createCheckpointSymlink(comm, path, "plugin." + name, "txt", checkpointId);
}

void SimulationRadialVelocityControl::restart(__UNUSED MPI_Comm comm, std::string path)
{
    auto filename = createCheckpointName(path, "plugin." + name, "txt");
    auto good = TextIO::read(filename, pid);
    if (!good) die("failed to read '%s'\n", filename.c_str());
}




PostprocessRadialVelocityControl::PostprocessRadialVelocityControl(std::string name, std::string filename) :
    PostprocessPlugin(name)
{
    auto status = fdump.open(filename, "w");
    if (status != FileWrapper::Status::Success)
        die("Could not open file '%s'", filename.c_str());
    fprintf(fdump.get(), "# time time_step velocity*r force/r**3\n");
}

void PostprocessRadialVelocityControl::deserialize(__UNUSED MPI_Status& stat)
{
    MirState::StepType currentTimeStep;
    MirState::TimeType currentTime;
    float vel, force;

    SimpleSerializer::deserialize(data, currentTime, currentTimeStep, vel, force);

    if (rank == 0)
    {
        fprintf(fdump.get(), "%g %lld %g %g\n", currentTime, currentTimeStep, vel, force);        
        fflush(fdump.get());
    }
}
