#include "anchor_particle.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>
#include <core/utils/kernel_launch.h>

namespace AnchorParticlesKernels
{

__global__ void anchorParticles(PVview view, int n, const int *pids, const float3 *poss, const float3 *vels, double3 *forces)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    int pid = pids[i];
    
    auto p = view.readParticle(pid);
    p.r = poss[i];
    p.u = vels[i];
    view.writeParticle(pid, p);

    auto f = view.forces[pid];
    forces[i] += make_double3(f.x, f.y, f.z);
}

} // namespace AnchorParticleKernels

AnchorParticlesPlugin::AnchorParticlesPlugin(const MirState *state, std::string name, std::string pvName,
                                             FuncTime3D positions, FuncTime3D velocities,
                                             std::vector<int> pids, int reportEvery) :
    SimulationPlugin(state, name),
    pvName(pvName),
    positions(positions),
    velocities(velocities),
    reportEvery(reportEvery)
{
    auto n = pids.size();
    
    this->pids.resize_anew(n);
    
    for (int i = 0; i < n; ++i)
    {
        int pid = pids[i];
        if (pid < 0)
            die("invalid particle id %d\n", pid);
        this->pids[i] = pid;
    }
    
    if (positions(0).size() != n)
        die("pids and positions must have the same size");

    if (velocities(0).size() != n)
        die("pids and velocities must have the same size");

    forces   .resize_anew(n);
    posBuffer.resize_anew(n);
    velBuffer.resize_anew(n);

    this->pids.uploadToDevice(defaultStream);
}

void AnchorParticlesPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    forces.clear(defaultStream);
}

void AnchorParticlesPlugin::afterIntegration(cudaStream_t stream)
{
    PVview view(pv, pv->local());

    int n = pids.size();
    const int nthreads = 32;
    const int nblocks = getNblocks(pids.size(), nthreads);

    if (view.size == 0) return;

    float t = (float) state->currentTime;
    const auto& domain = state->domain;

    auto poss = positions(t);
    auto vels = velocities(t);

    for (int i = 0; i < n; ++i)
    {
        posBuffer[i] = domain.global2local(poss[i]);
        velBuffer[i] = vels[i];
    }

    posBuffer.uploadToDevice(stream);
    velBuffer.uploadToDevice(stream);
    
    SAFE_KERNEL_LAUNCH(
            AnchorParticlesKernels::anchorParticles,
            nblocks, nthreads, 0, stream,
            view, n, pids.devPtr(), posBuffer.devPtr(), velBuffer.devPtr(), forces.devPtr() );

    ++ nsamples;
}

void AnchorParticlesPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer, pvName);
    send(sendBuffer);
}

void AnchorParticlesPlugin::serializeAndSend(cudaStream_t stream)
{
    if (!isTimeEvery(state, reportEvery)) return;

    forces.downloadFromDevice(stream);

    waitPrevSend();

    SimpleSerializer::serialize(sendBuffer, state->currentTime, nsamples, forces);
    send(sendBuffer);

    nsamples = 0;
    forces.clearDevice(stream);
}


AnchorParticlesStatsPlugin::AnchorParticlesStatsPlugin(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(makePath(path))
{}

AnchorParticlesStatsPlugin::~AnchorParticlesStatsPlugin()
{
    if (fout) fclose(fout);
}

void AnchorParticlesStatsPlugin::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void AnchorParticlesStatsPlugin::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string pvName;
    SimpleSerializer::deserialize(data, pvName);

    if (activated && rank == 0)
        fout = fopen( (path + pvName + ".txt").c_str(), "w" );
}

void AnchorParticlesStatsPlugin::deserialize(MPI_Status& stat)
{
    std::vector<double3> forces;
    MirState::TimeType currentTime;
    int nsamples;

    SimpleSerializer::deserialize(data, currentTime, nsamples, forces);

    MPI_Check( MPI_Reduce( (rank == 0 ? MPI_IN_PLACE : forces.data()),  forces.data(),  3 * forces.size(),  MPI_DOUBLE, MPI_SUM, 0, comm) );

    if (activated && rank == 0)
    {
        fprintf(fout, "%f", currentTime);
        for (auto& f : forces)
        {
            f /= nsamples;
            fprintf(fout, " %f %f %f", f.x, f.y, f.z);
        }
        fprintf(fout, "\n");
        fflush(fout);
    }
}
