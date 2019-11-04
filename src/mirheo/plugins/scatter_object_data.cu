#include "scatter_object_data.h"

#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{
namespace ScatterObjectDataKernels
{

template <typename T>
__global__ void copyObjectDataToParticles(int objSize, int nObjects, const T *srcObjData, T *dstParticleData)
{
    const int pid   = threadIdx.x + blockIdx.x * blockDim.x;
    const int objId = pid / objSize;

    if (objId >= nObjects) return;

    dstParticleData[pid] = srcObjData[objId];
}

} // namespace ScatterObjectDataKernels

ScatterObjectDataPlugin::ScatterObjectDataPlugin(const MirState *state, std::string name, std::string ovName,
                                                 std::string channelName, std::string savedName) :
    SimulationPlugin(state, name),
    ovName(ovName),
    channelName(channelName),
    savedName(savedName)
{}

void ScatterObjectDataPlugin::beforeForces(cudaStream_t stream)
{
    auto lov = ov->local();

    const auto& srcDesc = lov->dataPerObject  .getChannelDescOrDie(channelName);
    const auto& dstDesc = lov->dataPerParticle.getChannelDescOrDie(savedName);

    mpark::visit([&](auto srcBufferPtr)
    {
        auto dstBufferPtr = mpark::get<decltype(srcBufferPtr)>(dstDesc.varDataPtr);
        
        const int objSize  = lov->objSize;
        const int nObjects = lov->nObjects;

        constexpr int nthreads = 128;
        const int nblocks = getNblocks(objSize * nObjects, nthreads);
    
        SAFE_KERNEL_LAUNCH(
            ScatterObjectDataKernels::copyObjectDataToParticles,
            nblocks, nthreads, 0, stream,
            objSize, nObjects, srcBufferPtr->devPtr(), dstBufferPtr->devPtr());
    }, srcDesc.varDataPtr);
}

void ScatterObjectDataPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    ov = simulation->getOVbyNameOrDie(ovName);
    auto lov = ov->local();
    
    const auto& desc = lov->dataPerObject.getChannelDescOrDie(channelName);

    if (lov->dataPerParticle.checkChannelExists(savedName))
        die("Plugin '%s': a particle channel with name '%s' is already in use. Please choose a different name",
            name.c_str(), savedName.c_str());

    mpark::visit([&](auto pinnedBufferPtr)
    {
        using T = typename std::remove_reference< decltype(*pinnedBufferPtr->hostPtr()) >::type;
        ov->requireDataPerParticle<T>(savedName, DataManager::PersistenceMode::None);
    }, desc.varDataPtr);
}


} // namespace mirheo
