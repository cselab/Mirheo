#include "anchor_particle.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>
#include <core/utils/kernel_launch.h>

namespace AnchorParticleKernels
{

__global__ void anchorParticle(PVview view, int pid, float3 pos, float3 vel, double3 *force)
{
    if (threadIdx.x > 0) return;

    Particle p = view.readParticle(pid);
    p.r = pos;
    p.u = vel;
    view.writeParticle(pid, p);

    auto f = view.forces[pid];
    *force += make_double3(f.x, f.y, f.z);
}

} // namespace AnchorParticleKernels

AnchorParticlePlugin::AnchorParticlePlugin(const YmrState *state, std::string name, std::string pvName,
                                           FuncTime3D position, FuncTime3D velocity, int pid, int reportEvery) :
    SimulationPlugin(state, name),
    pvName(pvName),
    position(position),
    velocity(velocity),
    pid(pid),
    reportEvery(reportEvery)
{
    if (pid < 0)
        die("invalid particle id %d\n", pid);
}

void AnchorParticlePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);
    force.clear(defaultStream);
}

void AnchorParticlePlugin::afterIntegration(cudaStream_t stream)
{
    PVview view(pv, pv->local());
    const int nthreads = 32;
    const int nblocks = 1;

    if (view.size == 0) return;
    if (pid >= view.size) return;

    float t = (float) state->currentTime;
    const auto& domain = state->domain;

    float3 pos = position(t);
    float3 vel = velocity(t);
    pos = domain.global2local(pos);

    SAFE_KERNEL_LAUNCH(
            AnchorParticleKernels::anchorParticle,
            nblocks, nthreads, 0, stream,
            view, pid, pos, vel, force.devPtr() );

    ++ nsamples;
}

void AnchorParticlePlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer, pvName);
    send(sendBuffer);
}

void AnchorParticlePlugin::serializeAndSend(cudaStream_t stream)
{
    if (!isTimeEvery(state, reportEvery)) return;

    force.downloadFromDevice(stream);

    waitPrevSend();

    SimpleSerializer::serialize(sendBuffer, state->currentTime, nsamples, force[0]);
    send(sendBuffer);

    nsamples = 0;
    force.clearDevice(stream);
}


AnchorParticleStatsPlugin::AnchorParticleStatsPlugin(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(path)
{}

AnchorParticleStatsPlugin::~AnchorParticleStatsPlugin()
{
    if (fout) fclose(fout);
}

void AnchorParticleStatsPlugin::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void AnchorParticleStatsPlugin::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string pvName;
    SimpleSerializer::deserialize(data, pvName);

    if (activated && rank == 0)
        fout = fopen( (path + "/" + pvName + ".txt").c_str(), "w" );
}

void AnchorParticleStatsPlugin::deserialize(MPI_Status& stat)
{
    double3 force;
    YmrState::TimeType currentTime;
    int nsamples;

    SimpleSerializer::deserialize(data, currentTime, nsamples, force);

    MPI_Check( MPI_Reduce( (rank == 0 ? MPI_IN_PLACE : &force),  &force,  3,  MPI_DOUBLE, MPI_SUM, 0, comm) );

    if (activated && rank == 0)
    {
        force /= nsamples;
        fprintf(fout, "%f %f %f %f\n", currentTime, force.x, force.y, force.z);
        fflush(fout);
    }
}
