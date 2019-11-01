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
    pvName(pvName),
    dumpEvery(dumpEvery),
    region(state, "field_"+name, func, h)
{}

VirialPressurePlugin::~VirialPressurePlugin() = default;

void VirialPressurePlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    region.setup(comm);

    info("Plugin %s initialized for the following particle vector: %s", name.c_str(), pvName.c_str());
}

void VirialPressurePlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer, pvName);
    send(sendBuffer);
}

void VirialPressurePlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    PVview view(pv, pv->local());
    const Stress *stress = pv->local()->dataPerParticle.getData<Stress>(ChannelNames::stresses)->devPtr();

    localVirialPressure.clear(stream);

    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);
    
    SAFE_KERNEL_LAUNCH(
        VirialPressureKernels::totalPressure,
        nblocks, nthreads, 0, stream,
        view, stress, region.handler(), localVirialPressure.devPtr() );

    localVirialPressure.downloadFromDevice(stream, ContainersSynch::Synch);
    
    savedTime = state->currentTime;
    needToSend = true;
}

void VirialPressurePlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend) return;

    debug2("Plugin %s is sending now data", name.c_str());

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, savedTime, localVirialPressure[0]);
    send(sendBuffer);
    
    needToSend = false;
}

//=================================================================================

VirialPressureDumper::VirialPressureDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(makePath(path))
{}

void VirialPressureDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void VirialPressureDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string pvName;
    SimpleSerializer::deserialize(data, pvName);

    if (activated)
    {
        auto fname = path + pvName + ".txt";
        auto status = fdump.open(fname, "w");
        if (status != FileWrapper::Status::Success)
            die("Could not open file '%s'", fname.c_str());
        fprintf(fdump.get(), "# time Pressure\n");
    }
}

void VirialPressureDumper::deserialize()
{
    MirState::TimeType curTime;
    VirialPressure::ReductionType localPressure, totalPressure;

    SimpleSerializer::deserialize(data, curTime, localPressure);

    if (!activated) return;

    const auto dataType = getMPIFloatType<VirialPressure::ReductionType>();
    MPI_Check( MPI_Reduce(&localPressure, &totalPressure, 1, dataType, MPI_SUM, 0, comm) );

    fprintf(fdump.get(), "%g %.6e\n", curTime, totalPressure);
}

