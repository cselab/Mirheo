#include "wall_force_collector.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/walls/interface.h>

namespace WallForceCollector
{
__global__ void totalForce(PVview view, double3 *totalForce)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float3 f {0.f, 0.f, 0.f};
    
    if (tid < view.size)
        f = make_float3(view.forces[tid]);

    f = warpReduce(f, [](float a, float b) { return a + b; });

    if (__laneid() == 0)
        atomicAdd(totalForce, make_double3(f));
}
} //namespace WallForceCollector


WallForceCollectorPlugin::WallForceCollectorPlugin(const YmrState *state, std::string name,
                                                   std::string wallName, std::string frozenPvName,
                                                   int sampleEvery, int dumpEvery) :
    SimulationPlugin(state, name),
    sampleEvery(sampleEvery),
    dumpEvery(dumpEvery),
    wallName(wallName),
    frozenPvName(frozenPvName)
{}

WallForceCollectorPlugin::~WallForceCollectorPlugin() = default;


void WallForceCollectorPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    wall = dynamic_cast<SDF_basedWall*>(simulation->getWallByNameOrDie(wallName));

    if (wall == nullptr)
        die("Plugin '%s' expects a SDF based wall (got '%s')\n", name.c_str(), wallName.c_str());

    pv = simulation->getPVbyNameOrDie(frozenPvName);

    bounceForceBuffer = wall->getCurrentBounceForce();
}

void WallForceCollectorPlugin::afterIntegration(cudaStream_t stream)
{   
    if (isTimeEvery(state, sampleEvery))
    {
        pvForceBuffer.clear(stream);

        PVview view(pv, pv->local());
        const int nthreads = 128;

        SAFE_KERNEL_LAUNCH(
            WallForceCollector::totalForce,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, pvForceBuffer.devPtr() );

        pvForceBuffer     .downloadFromDevice(stream);
        bounceForceBuffer->downloadFromDevice(stream);

        totalForce += pvForceBuffer[0];
        totalForce += (*bounceForceBuffer)[0];

        ++nsamples;
    }
    
    needToDump = (isTimeEvery(state, dumpEvery) && nsamples > 0);
}

void WallForceCollectorPlugin::serializeAndSend(cudaStream_t stream)
{
    if (needToDump)
    {
        waitPrevSend();
        SimpleSerializer::serialize(sendBuffer, state->currentTime, nsamples, totalForce);
        send(sendBuffer);
        needToDump = false;
        nsamples   = 0;
        totalForce = make_double3(0, 0, 0);
    }
}

WallForceDumperPlugin::WallForceDumperPlugin(std::string name, std::string filename) :
    PostprocessPlugin(name)
{
    fdump = fopen(filename.c_str(), "w");
    if (!fdump)
        die("Could not open file '%s'", filename.c_str());
}

WallForceDumperPlugin::~WallForceDumperPlugin()
{
    if (fdump != nullptr) fclose(fdump);
}

void WallForceDumperPlugin::deserialize(MPI_Status& stat)
{
    YmrState::TimeType currentTime;
    int nsamples;
    double localForce[3], totalForce[3] = {0.0, 0.0, 0.0};

    SimpleSerializer::deserialize(data, currentTime, nsamples, localForce);
    
    MPI_Check( MPI_Reduce(localForce, totalForce, 3, MPI_DOUBLE, MPI_SUM, 0, comm) );

    if (rank == 0)
    {
        totalForce[0] /= (double)nsamples;
        totalForce[1] /= (double)nsamples;
        totalForce[2] /= (double)nsamples;

        if (fdump != nullptr) {
            fprintf(fdump, "%g %g %g %g\n", currentTime, totalForce[0], totalForce[1], totalForce[2]);
            fflush(fdump);
        }
    }
}


