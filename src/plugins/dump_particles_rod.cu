#include "dump_particles_rod.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/pvs/rod_vector.h>
#include <core/simulation.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/cuda_common.h>

namespace DumpParticlesRodKernels
{

template <typename T>
__global__ void copyRodQuantities(int numBiSegmentsPerObject, int objSize, int nObjects, const T *rodData, T *particleData)
{
    constexpr int stride = 5;
    int pid = threadIdx.x + blockIdx.x * blockDim.x;

    int objId        = pid / objSize;
    int localPartId  = pid % objSize;
    int localBisegId = min(localPartId / stride, numBiSegmentsPerObject); // min because of last particle

    int bid = objId * numBiSegmentsPerObject + localBisegId;

    if (objId < nObjects)
        particleData[pid] = rodData[bid];
}

} // namespace DumpParticlesRodKernels


ParticleWithRodQuantitiesSenderPlugin::
ParticleWithRodQuantitiesSenderPlugin(const YmrState *state, std::string name, std::string pvName, int dumpEvery,
                                      std::vector<std::string> channelNames,
                                      std::vector<ChannelType> channelTypes) :
    ParticleSenderPlugin(state, name, pvName, dumpEvery, channelNames, channelTypes)
{}

void ParticleWithRodQuantitiesSenderPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    rv = dynamic_cast<RodVector*>(pv);

    info("Plugin %s initialized for the following particle vector: %s", name.c_str(), pvName.c_str());
}

void ParticleWithRodQuantitiesSenderPlugin::beforeForces(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    positions .genericCopy(&pv->local()->positions() , stream);
    velocities.genericCopy(&pv->local()->velocities(), stream);

    auto& partManager  = pv->local()->dataPerParticle;
    auto& bisegManager = rv->local()->dataPerBisegment;

    for (int i = 0; i < channelNames.size(); ++i)
    {
        auto name = channelNames[i];
        if (partManager.checkChannelExists(name))
        {
            auto srcContainer = partManager.getGenericData(name);
            channelData[i].genericCopy(srcContainer, stream);
        }
        else
        {
            auto& desc = bisegManager.getChannelDescOrDie(name);
            auto& partData = channelRodData[name];
            
            mpark::visit([&](auto srcPinnedBuffer)
            {
                using Type = typename std::remove_pointer<decltype(srcPinnedBuffer)>::type::value_type;

                int nparticles = rv->local()->size();
                int objSize  = rv->objSize;
                int nObjects = rv->local()->nObjects;
                
                size_t sizeFloats = pv->local()->size() * sizeof(Type) / sizeof(float);
                partData.resize_anew(sizeFloats);

                const int nthreads = 128;
                const int nblocks = getNblocks(nparticles, nthreads);
                
                SAFE_KERNEL_LAUNCH(
                    DumpParticlesRodKernels::copyRodQuantities,
                    nblocks, nthreads, 0, stream,
                    rv->local()->getNumSegmentsPerRod(), objSize, nObjects,
                    srcPinnedBuffer->devPtr(), reinterpret_cast<Type*>(partData.devPtr()));
            }, desc.varDataPtr);

            channelData[i].genericCopy(&partData, stream);
        }
    }
}
