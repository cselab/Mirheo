#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/simulation.h>
#include <core/utils/folders.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include "virial_pressure.h"
#include "simple_serializer.h"

namespace VirialPressure
{
__global__ void totalPressure(int n, const Stress *stress, ReductionType *pressure)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int wid = tid % warpSize;
    if (tid >= n) return;

    const Stress s = stress[tid];

    ReductionType P = (s.xx + s.yy + s.zz) / 3.0;

    P = warpReduce(P, [](ReductionType a, ReductionType b) { return a+b; });

    if (wid == 0)
        atomicAdd(pressure, P);
}
}

VirialPressurePlugin::VirialPressurePlugin(const YmrState *state, std::string name, std::string pvName, std::string stressName, int dumpEvery) :
    SimulationPlugin(state, name),
    pvName(pvName),
    stressName(stressName),
    dumpEvery(dumpEvery)
{}

VirialPressurePlugin::~VirialPressurePlugin() = default;

void VirialPressurePlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    info("Plugin %s initialized for the following particle vector: %s", name.c_str(), pvName.c_str());
}

void VirialPressurePlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer, pvName);
    send(sendBuffer);
}

void VirialPressurePlugin::afterIntegration(cudaStream_t stream)
{
    if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

    int n = pv->local()->size();
    const Stress *stress = pv->local()->extraPerParticle.getData<Stress>(stressName)->devPtr();

    localVirialPressure.clear(stream);
    
    SAFE_KERNEL_LAUNCH(
        VirialPressure::totalPressure,
        getNblocks(n, 128), 128, 0, stream,
        n, stress, localVirialPressure.devPtr() );
    
    savedTime = currentTime;
    needToSend = true;
}

void VirialPressurePlugin::serializeAndSend(cudaStream_t stream)
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
    path(path)
{
    if (std::is_same<VirialPressure::ReductionType, float>::value)
        mpiReductionType = MPI_FLOAT;
    else if (std::is_same<VirialPressure::ReductionType, double>::value)
        mpiReductionType = MPI_DOUBLE;
    else
        die("Incompatible type");
}

VirialPressureDumper::~VirialPressureDumper()
{
    if (activated)
        fclose(fdump);
}

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
        auto fname = path + "/" + pvName + ".txt";
        fdump = fopen(fname.c_str(), "w");
        if (!fdump) die("Could not open file '%s'", fname.c_str());
        fprintf(fdump, "# time Pressure\n");
    }
}

void VirialPressureDumper::deserialize(MPI_Status& stat)
{
    float curTime;
    VirialPressure::ReductionType localPressure, totalPressure;

    SimpleSerializer::deserialize(data, curTime, localPressure);

    if (!activated) return;

    MPI_Check( MPI_Reduce(&localPressure, &totalPressure, 1, mpiReductionType, MPI_SUM, 0, comm) );

    fprintf(fdump, "%g %g\n", curTime, totalPressure);
}

