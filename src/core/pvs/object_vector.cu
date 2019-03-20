#include "object_vector.h"
#include "views/ov.h"

#include <core/utils/kernel_launch.h>
#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>
#include <core/xdmf/xdmf.h>

#include "restart_helpers.h"

namespace ObjectVectorKernels
{

__global__ void minMaxCom(OVview ovView)
{
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;
    const int objId = gid >> 5;
    const int tid = gid & 0x1f;
    if (objId >= ovView.nObjects) return;

    float3 mymin = make_float3( 1e+10f);
    float3 mymax = make_float3(-1e+10f);
    float3 mycom = make_float3(0);

#pragma unroll 3
    for (int i = tid; i < ovView.objSize; i += warpSize)
    {
        const int offset = (objId * ovView.objSize + i) * 2;

        const float3 coo = make_float3(ovView.particles[offset]);

        mymin = fminf(mymin, coo);
        mymax = fmaxf(mymax, coo);
        mycom += coo;
    }

    mycom = warpReduce( mycom, [] (float a, float b) { return a+b; } );
    mymin = warpReduce( mymin, [] (float a, float b) { return fmin(a, b); } );
    mymax = warpReduce( mymax, [] (float a, float b) { return fmax(a, b); } );

    if (tid == 0)
        ovView.comAndExtents[objId] = {mycom / ovView.objSize, mymin, mymax};
}

} // namespace ObjectVectorKernels

void ObjectVector::findExtentAndCOM(cudaStream_t stream, ParticleVectorType type)
{
    bool isLocal = (type == ParticleVectorType::Local);
    auto lov = isLocal ? local() : halo();

    if (lov->comExtentValid)
    {
        debug("COM and extent computation for %s OV '%s' skipped",
              isLocal ? "local" : "halo", name.c_str());
        return;
    }

    debug("Computing COM and extent OV '%s' (%s)", name.c_str(), isLocal ? "local" : "halo");

    const int nthreads = 128;
    OVview ovView(this, lov);
    SAFE_KERNEL_LAUNCH(
            ObjectVectorKernels::minMaxCom,
            (ovView.nObjects*32 + nthreads-1)/nthreads, nthreads, 0, stream,
            ovView );
}

void ObjectVector::_getRestartExchangeMap(MPI_Comm comm, const std::vector<Particle> &parts, std::vector<int>& map)
{
    int dims[3], periods[3], coords[3];
    MPI_Check( MPI_Cart_get(comm, 3, dims, periods, coords) );

    int nObjs = parts.size() / objSize;
    map.resize(nObjs);
    
    for (int i = 0, k = 0; i < nObjs; ++i) {
        auto com = make_float3(0);

        for (int j = 0; j < objSize; ++j, ++k)
            com += parts[k].r;

        com /= objSize;

        int3 procId3 = make_int3(floorf(com / state->domain.localSize));

        if (procId3.x >= dims[0] || procId3.y >= dims[1] || procId3.z >= dims[2]) {
            map[i] = -1;
            continue;
        }
        
        int procId;
        MPI_Check( MPI_Cart_rank(comm, (int*)&procId3, &procId) );
        map[i] = procId;
    }
}


std::vector<int> ObjectVector::_restartParticleData(MPI_Comm comm, std::string path)
{
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointName(path, "PV", "xmf");
    info("Restarting object vector %s from file %s", name.c_str(), filename.c_str());

    XDMF::readParticleData(filename, comm, this, objSize);

    std::vector<Particle> parts(local()->size());
    std::copy(local()->coosvels.begin(), local()->coosvels.end(), parts.begin());
    std::vector<int> map;
    
    _getRestartExchangeMap(comm, parts, map);
    RestartHelpers::exchangeData(comm, map, parts, objSize);    
    RestartHelpers::copyShiftCoordinates(state->domain, parts, local());

    local()->coosvels.uploadToDevice(defaultStream);
    
    // Do the ids
    // That's a kinda hack, will be properly fixed in the hdf5 per object restarts
    auto ids = local()->extraPerObject.getData<int>(ChannelNames::globalIds);
    for (int i = 0; i < local()->nObjects; i++)
        (*ids)[i] = local()->coosvels[i*objSize].i1 / objSize;
    ids->uploadToDevice(defaultStream);

    CUDA_Check( cudaDeviceSynchronize() );

    info("Successfully read %d particles", local()->coosvels.size());

    return map;
}

static void splitCom(DomainInfo domain, const PinnedBuffer<LocalObjectVector::COMandExtent>& com_extents, std::vector<float> &positions)
{
    int n = com_extents.size();
    positions.resize(3 * n);

    float3 *pos = (float3*) positions.data();
    
    for (int i = 0; i < n; ++i) {
        auto r = com_extents[i].com;
        pos[i] = domain.local2global(r);
    }
}

void ObjectVector::_extractPersistentExtraObjectData(std::vector<XDMF::Channel>& channels, const std::set<std::string>& blackList)
{
    auto& extraData = local()->extraPerObject;
    _extractPersistentExtraData(extraData, channels, blackList);
}

void ObjectVector::_checkpointObjectData(MPI_Comm comm, std::string path)
{
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointNameWithId(path, "OV", "");
    info("Checkpoint for object vector '%s', writing to file %s", name.c_str(), filename.c_str());

    auto coms_extents = local()->extraPerObject.getData<LocalObjectVector::COMandExtent>(ChannelNames::comExtents);

    coms_extents->downloadFromDevice(0, ContainersSynch::Synch);
    
    auto positions = std::make_shared<std::vector<float>>();

    splitCom(state->domain, *coms_extents, *positions);

    XDMF::VertexGrid grid(positions, comm);

    std::vector<XDMF::Channel> channels;

    _extractPersistentExtraObjectData(channels);
    
    XDMF::write(filename, &grid, channels, comm);

    createCheckpointSymlink(comm, path, "OV", "xmf");

    debug("Checkpoint for object vector '%s' successfully written", name.c_str());
}

void ObjectVector::_restartObjectData(MPI_Comm comm, std::string path, const std::vector<int>& map)
{
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointName(path, "OV", "xmf");
    info("Restarting object vector %s from file %s", name.c_str(), filename.c_str());

    XDMF::readObjectData(filename, comm, this);

    auto loc_ids = local()->extraPerObject.getData<int>(ChannelNames::globalIds);
    
    std::vector<int> ids(loc_ids->size());
    std::copy(loc_ids->begin(), loc_ids->end(), ids.begin());
    
    RestartHelpers::exchangeData(comm, map, ids, 1);

    loc_ids->resize_anew(ids.size());
    std::copy(ids.begin(), ids.end(), loc_ids->begin());

    loc_ids->uploadToDevice(defaultStream);
    CUDA_Check( cudaDeviceSynchronize() );

    info("Successfully read %d object infos", loc_ids->size());
}

void ObjectVector::checkpoint(MPI_Comm comm, std::string path)
{
    _checkpointParticleData(comm, path);
    _checkpointObjectData(comm, path);
}

void ObjectVector::restart(MPI_Comm comm, std::string path)
{
    auto map = _restartParticleData(comm, path);
    _restartObjectData(comm, path, map);
}
