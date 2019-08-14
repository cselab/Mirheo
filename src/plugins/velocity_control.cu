#include <core/utils/restart_helpers.h>

#include "velocity_control.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace VelocityControlKernels
{

inline __device__ bool is_inside(float3 r, float3 low, float3 high)
{
    return
        low.x <= r.x && r.x <= high.x &&
        low.y <= r.y && r.y <= high.y &&
        low.z <= r.z && r.z <= high.z;
}

__global__ void addForce(PVview view, DomainInfo domain, float3 low, float3 high, float3 force)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    auto r = Float3_int(view.readPosition(gid)).v;
    
    float3 gr = domain.local2global(r);

    if (is_inside(gr, low, high))
        view.forces[gid] += make_float4(force, 0.0f);
}

__global__ void sumVelocity(PVview view, DomainInfo domain, float3 low, float3 high, float3 *totVel, int *nSamples)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    Particle p;
    
    p.u = make_float3(0.0f);

    if (gid < view.size) {

        p = view.readParticle(gid);
        float3 gr = domain.local2global(p.r);

        if (is_inside(gr, low, high))
            atomicAggInc(nSamples);
        else
            p.u = make_float3(0.0f);
    }

    float3 u = warpReduce(p.u, [](float a, float b) { return a+b; });
    
    if (laneId() == 0 && dot(u, u) > 1e-8)
        atomicAdd(totVel, u);
}

} // namespace VelocityControlKernels

SimulationVelocityControl::SimulationVelocityControl(const MirState *state, std::string name, std::vector<std::string> pvNames,
                                                     float3 low, float3 high,
                                                     int sampleEvery, int tuneEvery, int dumpEvery,
                                                     float3 targetVel, float Kp, float Ki, float Kd) :
    SimulationPlugin(state, name),
    pvNames(pvNames),
    low(low),
    high(high),
    currentVel(make_float3(0,0,0)),
    targetVel(targetVel),
    sampleEvery(sampleEvery),
    tuneEvery(tuneEvery),
    dumpEvery(dumpEvery), 
    force(make_float3(0, 0, 0)),
    pid(make_float3(0, 0, 0), Kp, Ki, Kd),
    accumulatedTotVel({0,0,0})
{}


void SimulationVelocityControl::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    for (auto &pvName : pvNames)
        pvs.push_back(simulation->getPVbyNameOrDie(pvName));
}

void SimulationVelocityControl::beforeForces(cudaStream_t stream)
{
    for (auto &pv : pvs)
    {
        PVview view(pv, pv->local());
        const int nthreads = 128;

        SAFE_KERNEL_LAUNCH
            (VelocityControlKernels::addForce,
             getNblocks(view.size, nthreads), nthreads, 0, stream,
             view, state->domain, low, high, force );
    }
}

void SimulationVelocityControl::sampleOnePv(ParticleVector *pv, cudaStream_t stream) {
    PVview pvView(pv, pv->local());
    const int nthreads = 128;
 
    SAFE_KERNEL_LAUNCH
        (VelocityControlKernels::sumVelocity,
         getNblocks(pvView.size, nthreads), nthreads, 0, stream,
         pvView, state->domain, low, high, totVel.devPtr(), nSamples.devPtr());
}

void SimulationVelocityControl::afterIntegration(cudaStream_t stream)
{
    if (isTimeEvery(state, sampleEvery))
    {
        debug2("Velocity control %s is sampling now", name.c_str());

        totVel.clearDevice(stream);
        for (auto &pv : pvs) sampleOnePv(pv, stream);
        totVel.downloadFromDevice(stream);
        accumulatedTotVel.x += totVel[0].x;
        accumulatedTotVel.y += totVel[0].y;
        accumulatedTotVel.z += totVel[0].z;
    }
    
    if (!isTimeEvery(state, tuneEvery)) return;
    
    nSamples.downloadFromDevice(stream);
    nSamples.clearDevice(stream);
    
    long nSamples_loc, nSamples_tot = 0;
    double3 totVel_tot = make_double3(0,0,0);  

    nSamples_loc = nSamples[0];
    
    MPI_Check( MPI_Allreduce(&nSamples_loc,        &nSamples_tot, 1, MPI_LONG,   MPI_SUM, comm) );
    MPI_Check( MPI_Allreduce(&accumulatedTotVel,   &totVel_tot,   3, MPI_DOUBLE, MPI_SUM, comm) );

    currentVel = nSamples_tot ? make_float3(totVel_tot / nSamples_tot) : make_float3(0.f, 0.f, 0.f);
    force = pid.update(targetVel - currentVel);
    accumulatedTotVel = {0,0,0};
}

void SimulationVelocityControl::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, state->currentTime, state->currentStep, currentVel, force);
    send(sendBuffer);
}

void SimulationVelocityControl::checkpoint(MPI_Comm comm, std::string path, int checkpointId)
{
    auto filename = createCheckpointNameWithId(path, "plugin." + name, "txt", checkpointId);

    TextIO::write(filename, pid);
    
    createCheckpointSymlink(comm, path, "plugin." + name, "txt", checkpointId);
}

void SimulationVelocityControl::restart(__UNUSED MPI_Comm comm, std::string path)
{
    auto filename = createCheckpointName(path, "plugin." + name, "txt");
    auto good = TextIO::read(filename, pid);
    if (!good) die("failed to read '%s'\n", filename.c_str());
}




PostprocessVelocityControl::PostprocessVelocityControl(std::string name, std::string filename) :
    PostprocessPlugin(name)
{
    auto status = fdump.open(filename, "w");
    if (status != FileWrapper::Status::Success)
        die("Could not open file '%s'", filename.c_str());
    fprintf(fdump.get(), "# time time_step velocity force\n");
}

void PostprocessVelocityControl::deserialize()
{
    MirState::StepType currentTimeStep;
    MirState::TimeType currentTime;
    float3 vel, force;

    SimpleSerializer::deserialize(data, currentTime, currentTimeStep, vel, force);

    if (rank == 0) {
        fprintf(fdump.get(),
                "%g %lld "
                "%g %g %g "
                "%g %g %g\n",
                currentTime, currentTimeStep,
                vel.x, vel.y, vel.z,
                force.x, force.y, force.z);
        
        fflush(fdump.get());
    }
}
