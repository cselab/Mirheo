#include "xdmf.h"
#include "common.h"

#include "xmf_helpers.h"
#include "hdf5_helpers.h"

#include <core/logger.h>
#include <core/rigid_kernels/rigid_motion.h>
#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>
#include <core/utils/timer.h>

#include <hdf5.h>

namespace XDMF
{
void write(std::string filename, const Grid* grid, const std::vector<Channel>& channels, float time, MPI_Comm comm)
{        
    std::string h5Filename  = filename + ".h5";
    std::string xmfFilename = filename + ".xmf";
        
    info("Writing XDMF data to %s[.h5,.xmf]", filename.c_str());

    mTimer timer;
    timer.start();
    XMF::write(xmfFilename, relativePath(h5Filename), comm, grid, channels, time);
    HDF5::write(h5Filename, comm, grid, channels);
    info("Writing took %f ms", timer.elapsed());
}
    
void write(std::string filename, const Grid* grid, const std::vector<Channel>& channels, MPI_Comm comm)
{
    write(filename, grid, channels, -1, comm);
}

static long getLocalNumElements(const GridDims *gridDims)
{
    long n = 1;
    for (auto i : gridDims->getLocalSize())  n *= i;
    return n;
}

static void combineIntoPosVel(int n, const float3 *pos, const float3 *vel, const int64_t *ids,
                              float4 *outPos, float4 *outVel)
{
    for (int i = 0; i < n; ++i) {
        Particle p;
        p.r = pos[i];
        p.u = vel[i];
        p.setId(ids[i]);
        outPos[i] = p.r2Float4();
        outVel[i] = p.u2Float4();
    }    
}

static void addPersistentExtraDataPerParticle(int n, const Channel& channel, ParticleVector *pv)
{
    mpark::visit([&](auto typeWrapper)
    {
        using Type = typename decltype(typeWrapper)::type;

        pv->requireDataPerParticle<Type>
            (channel.name,
             DataManager::PersistenceMode::Active);

        auto buffer = pv->local()->dataPerParticle.getData<Type>(channel.name);
        buffer->resize_anew(n);
        memcpy(buffer->data(), channel.data, n * sizeof(Type));
        buffer->uploadToDevice(defaultStream);
    }, channel.type);
}
    
static void gatherFromChannels(const std::vector<Channel>& channels, const std::vector<float3>& positions, ParticleVector *pv)
{
    int n = positions.size();
    const float3 *vel = nullptr;
    const int64_t *ids = nullptr;

    pv->local()->resize_anew(n);
    auto& pos4 = pv->local()->positions();
    auto& vel4 = pv->local()->velocities();

    for (auto& ch : channels)
    {
        if      (ch.name == "velocity"              ) vel = (const float3*)  ch.data;            
        else if (ch.name == ChannelNames::globalIds ) ids = (const int64_t*) ch.data;
        else addPersistentExtraDataPerParticle(n, ch, pv);
    }

    if (n > 0 && vel == nullptr)
        die("Channel 'velocities' is required to read XDMF into a particle vector");
    if (n > 0 && ids == nullptr)
        die("Channel 'ids' is required to read XDMF into a particle vector");

    combineIntoPosVel(n, positions.data(), vel, ids, pos4.data(), vel4.data());

    pos4.uploadToDevice(defaultStream);
    vel4.uploadToDevice(defaultStream);
}

static void addPersistentExtraDataPerObject(int n, const Channel& channel, ObjectVector *ov)
{
    mpark::visit([&](auto typeWrapper)
    {
        using Type = typename decltype(typeWrapper)::type;
                     
        ov->requireDataPerObject<Type>
            (channel.name, DataManager::PersistenceMode::Active);
                     
        auto buffer = ov->local()->dataPerObject.getData<Type>(channel.name);
        buffer->resize_anew(n);
        memcpy(buffer->data(), channel.data, n * sizeof(Type));
        buffer->uploadToDevice(defaultStream);
    }, channel.type);
}
    
static void gatherFromChannels(const std::vector<Channel>& channels, const std::vector<float3>& positions, ObjectVector *ov)
{
    int n = positions.size();
    const int64_t *ids_data = nullptr;

    auto ids = ov->local()->dataPerObject.getData<int64_t>(ChannelNames::globalIds);

    ids->resize_anew(n);

    for (auto& ch : channels)
    {
        if (ch.name == ChannelNames::globalIds) ids_data = (const int64_t*) ch.data;
        else addPersistentExtraDataPerObject(n, ch, ov);
    }

    if (n > 0 && ids_data == nullptr)
        die("Channel '%s' is required to read XDMF into an object vector", ChannelNames::globalIds.c_str());

    for (int i = 0; i < n; ++i)
    {
        (*ids)[i] = ids_data[i];
    }

    ids->uploadToDevice(defaultStream);
}

struct RigidMotionsSoA
{
    const float3 *pos            {nullptr};
    const RigidReal4 *quaternion {nullptr};
    const RigidReal3 *vel        {nullptr};
    const RigidReal3 *omega      {nullptr};
    const RigidReal3 *force      {nullptr};
    const RigidReal3 *torque     {nullptr};    
};

static void combineIntoRigidMotions(int n, const RigidMotionsSoA& soa, RigidMotion *motions)
{
    for (int i = 0; i < n; ++i)
    {
        RigidMotion m;
        m.r      = make_rigidReal3(soa.pos[i]);
        m.q      = soa.quaternion[i];
        m.vel    = soa.vel       [i];
        m.omega  = soa.omega     [i];
        m.force  = soa.force     [i];
        m.torque = soa.torque    [i];
        motions[i] = m;
    }
}
    
static void gatherFromChannels(const std::vector<Channel>& channels, const std::vector<float3>& positions, RigidObjectVector *rov)
{
    int n = positions.size();
    const int64_t *ids_data {nullptr};
    RigidMotionsSoA soa;

    soa.pos = positions.data();

    auto ids     = rov->local()->dataPerObject.getData<int64_t>(ChannelNames::globalIds);
    auto motions = rov->local()->dataPerObject.getData<RigidMotion>(ChannelNames::motions);

    ids    ->resize_anew(n);
    motions->resize_anew(n);

    for (auto& ch : channels)
    {
        if      (ch.name == ChannelNames::globalIds)  ids_data       = reinterpret_cast<const int64_t*>    (ch.data);
        else if (ch.name == "quaternion")             soa.quaternion = reinterpret_cast<const RigidReal4*> (ch.data); 
        else if (ch.name == "velocity")               soa.vel        = reinterpret_cast<const RigidReal3*> (ch.data);
        else if (ch.name == "omega")                  soa.omega      = reinterpret_cast<const RigidReal3*> (ch.data);
        else if (ch.name == "force")                  soa.force      = reinterpret_cast<const RigidReal3*> (ch.data);
        else if (ch.name == "torque")                 soa.torque     = reinterpret_cast<const RigidReal3*> (ch.data);
        else addPersistentExtraDataPerObject(n, ch, rov);
    }

    if (n > 0)
    {
        auto check = [&](std::string name, const void *ptr)
        {
            if (ptr == nullptr)
                die("Channel '%s' is required to read XDMF into an object vector", name.c_str());
        };
        check(ChannelNames::globalIds, ids_data);
        check("quaternion",            soa.quaternion);
        check("velocity",              soa.vel);
        check("omega",                 soa.omega);
        check("force",                 soa.force);
        check("torque",                soa.torque);
    }

    combineIntoRigidMotions(n, soa, motions->data());        

    for (int i = 0; i < n; ++i)
        (*ids)[i] = ids_data[i];

    ids    ->uploadToDevice(defaultStream);
    motions->uploadToDevice(defaultStream);
}
    
template <typename PV>
static void readData(std::string filename, MPI_Comm comm, PV *pv, int chunkSize) 
{
    info("Reading XDMF data from %s", filename.c_str());

    std::string h5filename;
        
    auto positions = std::make_shared<std::vector<float3>>();
    std::vector<std::vector<char>> channelData;
    std::vector<Channel> channels;
        
    VertexGrid grid(positions, comm);

    mTimer timer;
    timer.start();
    XMF::read(filename, comm, h5filename, &grid, channels);
    grid.splitReadAccess(comm, chunkSize);

    h5filename = parentPath(filename) + h5filename;

    long nElements = getLocalNumElements(grid.getGridDims());
    channelData.resize(channels.size());        

    debug("Got %d channels with %d items each", channels.size(), nElements);

    for (int i = 0; i < channels.size(); ++i) {
        channelData[i].resize(nElements * channels[i].nComponents() * channels[i].precision());
        channels[i].data = channelData[i].data();
    }
        
    HDF5::read(h5filename, comm, &grid, channels);
    info("Reading took %f ms", timer.elapsed());

    gatherFromChannels(channels, *positions, pv);
}

void readParticleData(std::string filename, MPI_Comm comm, ParticleVector *pv, int chunkSize)
{
    readData(filename, comm, pv, chunkSize);
}

    
void readObjectData(std::string filename, MPI_Comm comm, ObjectVector *ov)
{
    readData(filename, comm, ov, 1);
}

void readRigidObjectData(std::string filename, MPI_Comm comm, RigidObjectVector *rov)
{
    readData(filename, comm, rov, 1);
}

} // namespace XDMF
