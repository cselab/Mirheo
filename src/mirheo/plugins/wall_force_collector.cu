// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "wall_force_collector.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/path.h>
#include <mirheo/core/walls/interface.h>

namespace mirheo
{

namespace wall_force_collector_kernels
{
__global__ void totalForce(PVview view, double3 *totalForce)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    real3 f {0._r, 0._r, 0._r};

    if (tid < view.size)
        f = make_real3(view.forces[tid]);

    f = warpReduce(f, [](real a, real b) { return a + b; });

    if (laneId() == 0)
        atomicAdd(totalForce, make_double3(f));
}
} //namespace wall_force_collector_kernels


WallForceCollectorPlugin::WallForceCollectorPlugin(const MirState *state, std::string name,
                                                   std::string wallName, std::string frozenPvName,
                                                   int sampleEvery, int dumpEvery) :
    SimulationPlugin(state, name),
    sampleEvery_(sampleEvery),
    dumpEvery_(dumpEvery),
    wallName_(wallName),
    frozenPvName_(frozenPvName)
{}

WallForceCollectorPlugin::~WallForceCollectorPlugin() = default;


void WallForceCollectorPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    wall_ = dynamic_cast<SDFBasedWall*>(simulation->getWallByNameOrDie(wallName_));

    if (wall_ == nullptr)
        die("Plugin '%s' expects a SDF based wall (got '%s')\n", getCName(), wallName_.c_str());

    pv_ = simulation->getPVbyNameOrDie(frozenPvName_);

    bounceForceBuffer_ = wall_->getCurrentBounceForce();
}

void WallForceCollectorPlugin::afterIntegration(cudaStream_t stream)
{
    if (isTimeEvery(getState(), sampleEvery_))
    {
        pvForceBuffer_.clear(stream);

        PVview view(pv_, pv_->local());
        const int nthreads = 128;

        SAFE_KERNEL_LAUNCH(
            wall_force_collector_kernels::totalForce,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, pvForceBuffer_.devPtr() );

        pvForceBuffer_     .downloadFromDevice(stream, ContainersSynch::Asynch);
        bounceForceBuffer_->downloadFromDevice(stream, ContainersSynch::Synch);

        pvForce_     += pvForceBuffer_[0];
        bounceForce_ += (*bounceForceBuffer_)[0];

        ++nsamples_;
    }

    needToDump_ = (isTimeEvery(getState(), dumpEvery_) && nsamples_ > 0);
}

void WallForceCollectorPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (needToDump_)
    {
        _waitPrevSend();
        SimpleSerializer::serialize(sendBuffer_, getState()->currentTime, nsamples_, bounceForce_, pvForce_);
        _send(sendBuffer_);
        needToDump_ = false;
        nsamples_   = 0;
        bounceForce_ = make_double3(0, 0, 0);
        pvForce_     = make_double3(0, 0, 0);
    }
}

WallForceDumperPlugin::WallForceDumperPlugin(std::string name, std::string filename, bool detailedDump) :
    PostprocessPlugin(name),
    detailedDump_(detailedDump)
{
    filename = setExtensionOrDie(filename, "csv");

    auto status = fdump_.open(filename, "w");
    if (status != FileWrapper::Status::Success)
        die("Could not open file '%s'", filename.c_str());

    if (detailedDump_)
        fprintf(fdump_.get(), "time,fbx,fby,fbz,fix,fiy,fiz\n");
    else
        fprintf(fdump_.get(), "time,fx,fy,fz\n");
}

void WallForceDumperPlugin::deserialize()
{
    MirState::TimeType currentTime;
    int nsamples;

    constexpr int ncomps = 6;
    double localForce[ncomps], totalForce[ncomps] = {0.0, 0.0, 0.0,  0.0, 0.0, 0.0};

    SimpleSerializer::deserialize(data_, currentTime, nsamples, localForce);

    MPI_Check( MPI_Reduce(localForce, totalForce, ncomps, MPI_DOUBLE, MPI_SUM, 0, comm_) );

    if (rank_ == 0)
    {
        for (int i = 0; i < ncomps; ++i)
            totalForce[i] /= (double)nsamples;

        const auto fb = make_double3(totalForce[0], totalForce[1], totalForce[2]);
        const auto fi = make_double3(totalForce[3], totalForce[4], totalForce[5]);

        if (detailedDump_)
        {
            fprintf(fdump_.get(), "%g,%g,%g,%g,%g,%g,%g\n",
                    currentTime,
                    fb.x, fb.y, fb.z,
                    fi.x, fi.y, fi.z);
        }
        else
        {
            const auto f = fb + fi;
            fprintf(fdump_.get(), "%g,%g,%g,%g\n", currentTime, f.x, f.y, f.z);
        }

        fflush(fdump_.get());
    }
}

} // namespace mirheo
