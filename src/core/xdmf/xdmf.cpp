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

static void combineIntoParticles(int n, const float3 *pos, const float3 *vel, const int *ids, Particle *particles)
{
    for (int i = 0; i < n; ++i) {
        Particle p;
        p.r = pos[i];
        p.u = vel[i];
        p.i1 = ids[i];
        p.i2 = 0;

        particles[i] = p;
    }    
}

static void addPersistentExtraDataPerParticle(int n, const Channel& channel, ParticleVector *pv)
{
    mpark::visit([&](auto typeWrapper) {
                     using Type = typename decltype(typeWrapper)::type;

                     pv->requireDataPerParticle<Type>
                         (channel.name,
                          ExtraDataManager::PersistenceMode::Persistent);
                     
                     auto buffer = pv->local()->extraPerParticle.getData<Type>(channel.name);
                     buffer->resize_anew(n);
                     memcpy(buffer->data(), channel.data, n * sizeof(Type));
                     buffer->uploadToDevice(defaultStream);
                 }, channel.type);
}
    
static void gatherFromChannels(std::vector<Channel> &channels, std::vector<float> &positions, ParticleVector *pv)
{
    int n = positions.size() / 3;
    const float3 *pos, *vel = nullptr;
    const int *ids = nullptr;

    pv->local()->resize_anew(n);
    auto& coosvels = pv->local()->coosvels;

    for (auto& ch : channels)
    {
        if      (ch.name == "velocity"              ) vel = (const float3*) ch.data;            
        else if (ch.name == ChannelNames::globalIds ) ids = (const int*) ch.data;
        else addPersistentExtraDataPerParticle(n, ch, pv);
    }

    if (n > 0 && vel == nullptr)
        die("Channel 'velocities' is required to read XDMF into a particle vector");
    if (n > 0 && ids == nullptr)
        die("Channel 'ids' is required to read XDMF into a particle vector");

    pos = (const float3*) positions.data();

    combineIntoParticles(n, pos, vel, ids, coosvels.data());

    coosvels.uploadToDevice(defaultStream);
}

static void addPersistentExtraDataPerObject(int n, const Channel& channel, ObjectVector *ov)
{
    mpark::visit([&](auto typeWrapper) {

                     using Type = typename decltype(typeWrapper)::type;
                     
                     ov->requireDataPerObject<Type>
                         (channel.name, ExtraDataManager::PersistenceMode::Persistent);
                     
                     auto buffer = ov->local()->extraPerObject.getData<Type>(channel.name);
                     buffer->resize_anew(n);
                     memcpy(buffer->data(), channel.data, n * sizeof(Type));
                     buffer->uploadToDevice(defaultStream);
                 }, channel.type);
}
    
static void gatherFromChannels(std::vector<Channel> &channels, std::vector<float> &positions, ObjectVector *ov)
{
    int n = positions.size() / 3;
    const int *ids_data = nullptr;

    auto ids = ov->local()->extraPerObject.getData<int>(ChannelNames::globalIds);

    ids->resize_anew(n);

    for (auto& ch : channels)
    {
        if (ch.name == ChannelNames::globalIds) ids_data = (const int*) ch.data;
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

static void combineIntoRigidMotions(int n, const float3 *pos, const RigidReal4 *quaternion,
                                    const RigidReal3 *vel, const RigidReal3 *omega,
                                    const RigidReal3 *force, const RigidReal3 *torque,
                                    RigidMotion *motions)
{
    for (int i = 0; i < n; ++i) {
        RigidMotion m;
        m.r      = make_rigidReal3(pos[i]);
        m.q      = quaternion[i];
        m.vel    = vel[i];
        m.omega  = omega[i];
        m.force  = force[i];
        m.torque = torque[i];
            
        motions[i] = m;
    }
}
    
static void gatherFromChannels(std::vector<Channel> &channels, std::vector<float> &positions, RigidObjectVector *rov)
{
    int n = positions.size() / 3;
    const int *ids_data = nullptr;
    const float3 *pos = (const float3*) positions.data();
    const RigidReal4 *quaternion;
    const RigidReal3 *vel, *omega, *force, *torque;

    auto ids     = rov->local()->extraPerObject.getData<int>(ChannelNames::globalIds);
    auto motions = rov->local()->extraPerObject.getData<RigidMotion>(ChannelNames::motions);

    ids    ->resize_anew(n);
    motions->resize_anew(n);

    for (auto& ch : channels)
    {
        if      (ch.name == ChannelNames::globalIds)  ids_data = (const int*)        ch.data;
        else if (ch.name == "quaternion") quaternion = (const RigidReal4*) ch.data; 
        else if (ch.name == "velocity")          vel = (const RigidReal3*) ch.data;
        else if (ch.name == "omega")           omega = (const RigidReal3*) ch.data;
        else if (ch.name == "force")           force = (const RigidReal3*) ch.data;
        else if (ch.name == "torque")         torque = (const RigidReal3*) ch.data;
        else addPersistentExtraDataPerObject(n, ch, rov);
    }

    if (n > 0) {
        auto check = [&](std::string name, const void *ptr) {
                         if (ptr == nullptr)
                             die("Channel '%s' is required to read XDMF into an object vector", name.c_str());
                     };
        check(ChannelNames::globalIds,        ids_data);
        check("quaternion", quaternion);
        check("velocity",   vel);
        check("omega",      omega);
        check("force",      force);
        check("torque",     torque);
    }

    combineIntoRigidMotions(n, pos, quaternion, vel, omega, force, torque, motions->data());        
    for (int i = 0; i < n; ++i) (*ids)[i] = ids_data[i];

    ids->uploadToDevice(defaultStream);
    motions->uploadToDevice(defaultStream);
}
    
template <typename PV>
static void readData(std::string filename, MPI_Comm comm, PV *pv, int chunk_size) 
{
    info("Reading XDMF data from %s", filename.c_str());

    std::string h5filename;
        
    auto positions = std::make_shared<std::vector<float>>();
    std::vector<std::vector<char>> channelData;
    std::vector<Channel> channels;
        
    VertexGrid grid(positions, comm);

    mTimer timer;
    timer.start();
    XMF::read(filename, comm, h5filename, &grid, channels);
    grid.split_read_access(comm, chunk_size);

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

void readParticleData(std::string filename, MPI_Comm comm, ParticleVector *pv, int chunk_size)
{
    readData(filename, comm, pv, chunk_size);
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
