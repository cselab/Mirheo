#include "virial_pressure.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/mpi_types.h>

namespace mirheo
{

namespace VirialPressureKernels
{
__global__ void totalPressure(PVview view, const Stress *stress, FieldDeviceHandler region, VirialPressure::ReductionType *pressure)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    VirialPressure::ReductionType P = 0;
    Particle p;

    if (tid < view.size) {
        const Stress s = stress[tid];
        auto r = Real3_int(view.readPosition(tid)).v;

        if (region(r) > 0)
            P = (s.xx + s.yy + s.zz) / 3.0;
    }
    
    P = warpReduce(P, [](VirialPressure::ReductionType a, VirialPressure::ReductionType b) { return a+b; });

    if (laneId() == 0)
        atomicAdd(pressure, P);
}
} // namespace VirialPressureKernels

VirialPressurePlugin::VirialPressurePlugin(const MirState *state, std::string name, std::string pvName,
                                           FieldFunction func, real3 h, int dumpEvery) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    dumpEvery_(dumpEvery),
    region_(state, "field_"+name, func, h)
{}

VirialPressurePlugin::~VirialPressurePlugin() = default;

void VirialPressurePlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    region_.setup(comm);

    info("Plugin %s initialized for the following particle vector: %s", getCName(), pvName_.c_str());
}

void VirialPressurePlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer_, pvName_);
    send(sendBuffer_);
}

void VirialPressurePlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    PVview view(pv_, pv_->local());
    const Stress *stress = pv_->local()->dataPerParticle.getData<Stress>(ChannelNames::stresses)->devPtr();

    localVirialPressure_.clear(stream);

    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);
    
    SAFE_KERNEL_LAUNCH(
        VirialPressureKernels::totalPressure,
        nblocks, nthreads, 0, stream,
        view, stress, region_.handler(), localVirialPressure_.devPtr() );

    localVirialPressure_.downloadFromDevice(stream, ContainersSynch::Synch);
    
    savedTime_ = getState()->currentTime;
    needToSend_ = true;
}

void VirialPressurePlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend_) return;

    debug2("Plugin %s is sending now data", getCName());

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, savedTime_, localVirialPressure_[0]);
    send(sendBuffer_);
    
    needToSend_ = false;
}

//=================================================================================

VirialPressureDumper::VirialPressureDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path))
{}

void VirialPressureDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void VirialPressureDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string pvName;
    SimpleSerializer::deserialize(data_, pvName);

    if (activated_)
    {
        auto fname = path_ + pvName + ".txt";
        auto status = fdump_.open(fname, "w");
        if (status != FileWrapper::Status::Success)
            die("Could not open file '%s'", fname.c_str());
        fprintf(fdump_.get(), "# time Pressure\n");
    }
}

void VirialPressureDumper::deserialize()
{
    MirState::TimeType curTime;
    VirialPressure::ReductionType localPressure, totalPressure;

    SimpleSerializer::deserialize(data_, curTime, localPressure);

    if (!activated_) return;

    const auto dataType = getMPIFloatType<VirialPressure::ReductionType>();
    MPI_Check( MPI_Reduce(&localPressure, &totalPressure, 1, dataType, MPI_SUM, 0, comm_) );

    fprintf(fdump_.get(), "%g %.6e\n", curTime, totalPressure);
}

} // namespace mirheo
