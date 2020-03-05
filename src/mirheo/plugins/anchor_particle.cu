#include "anchor_particle.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace anchor_particles_kernels
{

__global__ void anchorParticles(PVview view, int n, const int *pids, const real3 *poss, const real3 *vels, double3 *forces)
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

} // namespace anchor_particles_kernels

AnchorParticlesPlugin::AnchorParticlesPlugin(const MirState *state, std::string name, std::string pvName,
                                             FuncTime3D positions, FuncTime3D velocities,
                                             std::vector<int> pids, int reportEvery) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    positions_(positions),
    velocities_(velocities),
    reportEvery_(reportEvery)
{
    const size_t n = pids.size();
    
    pids_.resize_anew(n);
    
    for (size_t i = 0; i < n; ++i)
    {
        const auto pid = pids[i];
        if (pid < 0)
            die("invalid particle id %d\n", pid);
        pids_[i] = pid;
    }
    
    if (positions_(0).size() != n)
        die("pids and positions must have the same size");

    if (velocities_(0).size() != n)
        die("pids and velocities must have the same size");

    forces_   .resize_anew(n);
    posBuffer_.resize_anew(n);
    velBuffer_.resize_anew(n);

    pids_.uploadToDevice(defaultStream);
}

void AnchorParticlesPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    forces_.clear(defaultStream);
}

void AnchorParticlesPlugin::afterIntegration(cudaStream_t stream)
{
    PVview view(pv_, pv_->local());

    const int n = pids_.size();
    const int nthreads = 32;
    const int nblocks = getNblocks(n, nthreads);

    if (view.size == 0) return;

    const real t = (real) getState()->currentTime;
    const auto& domain = getState()->domain;

    auto poss = positions_(t);
    auto vels = velocities_(t);

    for (int i = 0; i < n; ++i)
    {
        posBuffer_[i] = domain.global2local(poss[i]);
        velBuffer_[i] = vels[i];
    }

    posBuffer_.uploadToDevice(stream);
    velBuffer_.uploadToDevice(stream);
    
    SAFE_KERNEL_LAUNCH(
            anchor_particles_kernels::anchorParticles,
            nblocks, nthreads, 0, stream,
            view, n, pids_.devPtr(), posBuffer_.devPtr(), velBuffer_.devPtr(), forces_.devPtr() );

    ++nsamples_;
}

void AnchorParticlesPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer_, pvName_);
    _send(sendBuffer_);
}

void AnchorParticlesPlugin::serializeAndSend(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), reportEvery_)) return;

    forces_.downloadFromDevice(stream);

    _waitPrevSend();

    SimpleSerializer::serialize(sendBuffer_, getState()->currentTime, nsamples_, forces_);
    _send(sendBuffer_);

    nsamples_ = 0;
    forces_.clearDevice(stream);
}


AnchorParticlesStatsPlugin::AnchorParticlesStatsPlugin(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path))
{}

void AnchorParticlesStatsPlugin::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void AnchorParticlesStatsPlugin::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string pvName;
    SimpleSerializer::deserialize(data_, pvName);

    if (activated_ && rank_ == 0)
    {
        const std::string fname = path_ + pvName + ".txt";
        auto status = fout_.open( fname, "w" );
        if (status != FileWrapper::Status::Success)
            die("could not open file '%s'", fname.c_str());
    }
}

void AnchorParticlesStatsPlugin::deserialize()
{
    std::vector<double3> forces;
    MirState::TimeType currentTime;
    int nsamples;

    SimpleSerializer::deserialize(data_, currentTime, nsamples, forces);

    MPI_Check( MPI_Reduce( (rank_ == 0 ? MPI_IN_PLACE : forces.data()),  forces.data(),  3 * forces.size(),  MPI_DOUBLE, MPI_SUM, 0, comm_) );

    if (activated_ && rank_ == 0)
    {
        fprintf(fout_.get(), "%f", currentTime);
        for (auto& f : forces)
        {
            f /= nsamples;
            fprintf(fout_.get(), " %f %f %f", f.x, f.y, f.z);
        }
        fprintf(fout_.get(), "\n");
        fflush(fout_.get());
    }
}

} // namespace mirheo
