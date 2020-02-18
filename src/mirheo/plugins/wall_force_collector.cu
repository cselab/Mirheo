#include "wall_force_collector.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/walls/interface.h>

namespace mirheo
{

namespace WallForceCollector
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
} //namespace WallForceCollector


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
            WallForceCollector::totalForce,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, pvForceBuffer_.devPtr() );

        pvForceBuffer_     .downloadFromDevice(stream);
        bounceForceBuffer_->downloadFromDevice(stream);

        totalForce_ += pvForceBuffer_[0];
        totalForce_ += (*bounceForceBuffer_)[0];

        ++nsamples_;
    }
    
    needToDump_ = (isTimeEvery(getState(), dumpEvery_) && nsamples_ > 0);
}

void WallForceCollectorPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (needToDump_)
    {
        waitPrevSend();
        SimpleSerializer::serialize(sendBuffer_, getState()->currentTime, nsamples_, totalForce_);
        send(sendBuffer_);
        needToDump_ = false;
        nsamples_   = 0;
        totalForce_ = make_double3(0, 0, 0);
    }
}

WallForceDumperPlugin::WallForceDumperPlugin(std::string name, std::string filename) :
    PostprocessPlugin(name)
{
    auto status = fdump_.open(filename, "w");
    if (status != FileWrapper::Status::Success)
        die("Could not open file '%s'", filename.c_str());
}

void WallForceDumperPlugin::deserialize()
{
    MirState::TimeType currentTime;
    int nsamples;
    double localForce[3], totalForce[3] = {0.0, 0.0, 0.0};

    SimpleSerializer::deserialize(data_, currentTime, nsamples, localForce);
    
    MPI_Check( MPI_Reduce(localForce, totalForce, 3, MPI_DOUBLE, MPI_SUM, 0, comm_) );

    if (rank_ == 0)
    {
        totalForce[0] /= (double)nsamples;
        totalForce[1] /= (double)nsamples;
        totalForce[2] /= (double)nsamples;

        fprintf(fdump_.get(), "%g %g %g %g\n",
                currentTime, totalForce[0], totalForce[1], totalForce[2]);
        fflush(fdump_.get());
    }
}

} // namespace mirheo
