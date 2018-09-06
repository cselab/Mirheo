#include "dump_particles.h"
#include "simple_serializer.h"
#include <core/utils/folders.h>

#include <core/simulation.h>
#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>
#include <core/utils/make_unique.h>

#include <regex>

ParticleSenderPlugin::ParticleSenderPlugin(std::string name, std::string pvName, int dumpEvery,
                                           std::vector<std::string> channelNames,
                                           std::vector<ChannelType> channelTypes) :
    SimulationPlugin(name), pvName(pvName),
    dumpEvery(dumpEvery), channelNames(channelNames), channelTypes(channelTypes)
{
    channelData.resize(channelNames.size());
}

void ParticleSenderPlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(sim, comm, interComm);

    pv = sim->getPVbyNameOrDie(pvName);

    info("Plugin %s initialized for the following particle vector: %s", name.c_str(), pvName.c_str());
}

void ParticleSenderPlugin::handshake()
{
    std::vector<int> sizes;

    for (auto t : channelTypes)
        switch (t) {
        case ChannelType::Scalar:
            sizes.push_back(1);
            break;
        case ChannelType::Vector:
            sizes.push_back(3);
            break;
        case ChannelType::Tensor6:
            sizes.push_back(6);
            break;
        }

    SimpleSerializer::serialize(sendBuffer, sim->nranks3D, sizes);

    int namesSize = 0;
    for (auto& s : channelNames)
        namesSize += SimpleSerializer::totSize(s);

    int shift = sendBuffer.size();
    sendBuffer.resize(sendBuffer.size() + namesSize);

    for (auto& s : channelNames) {
        SimpleSerializer::serialize(sendBuffer.data() + shift, s);
        shift += SimpleSerializer::totSize(s);
    }

    send(sendBuffer);
}

void ParticleSenderPlugin::beforeForces(cudaStream_t stream)
{
    if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

    particles.genericCopy(&pv->local()->coosvels, stream);

    for (int i = 0; i < channelNames.size(); ++i) {
        auto name = channelNames[i];
        auto srcContainer = pv->local()->extraPerParticle.getGenericData(name);
        channelData[i].genericCopy(srcContainer, stream); 
    }
}

void ParticleSenderPlugin::serializeAndSend(cudaStream_t stream)
{
    if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

    debug2("Plugin %s is sending now data", name.c_str());
    
    for (auto& p : particles)
        p.r = sim->domain.local2global(p.r);

    // Calculate total size for sending
    int totalSize = SimpleSerializer::totSize(currentTime, particles);
    for (auto& ch : channelData)
        totalSize += SimpleSerializer::totSize(ch);

    // Now allocate the sending buffer and pack everything into it
    debug2("Plugin %s is packing now data", name.c_str());
    sendBuffer.resize(totalSize);
    SimpleSerializer::serialize(sendBuffer.data(), currentTime, particles);
    int currentSize = SimpleSerializer::totSize(currentTime, particles);

    for (auto& ch : channelData) {
        SimpleSerializer::serialize(sendBuffer.data() + currentSize, ch);
        currentSize += SimpleSerializer::totSize(ch);
    }

    send(sendBuffer);
}




ParticleDumperPlugin::ParticleDumperPlugin(std::string name, std::string path) :
    PostprocessPlugin(name), path(path)
{}

void ParticleDumperPlugin::handshake()
{
    int3 nranks3D;
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::vector<int> sizes;
    SimpleSerializer::deserialize(data, nranks3D, sizes);

    std::vector<XDMFDumper::ChannelType> channelTypes;

    channelNames.push_back("velocity");
    channelTypes.push_back(XDMFDumper::ChannelType::Vector);
    
    for (auto s : sizes) {
        switch (s) {
        case 1: channelTypes.push_back(XDMFDumper::ChannelType::Scalar);  break;
        case 3: channelTypes.push_back(XDMFDumper::ChannelType::Vector);  break;
        case 6: channelTypes.push_back(XDMFDumper::ChannelType::Tensor6); break;
        default:
            die("Plugin '%s' got %d as a channel size, expected 1, 3 or 6", name.c_str(), s);
        }
    }

    // -1 because velocity will be a separate vector
    channelData.resize(channelTypes.size()-1);

    std::string allNames;
    int shift = SimpleSerializer::totSize(nranks3D, sizes);

    for (int i = 0; i < sizes.size(); ++i) {
        std::string s;
        SimpleSerializer::deserialize(data.data() + shift, s);
        channelNames.push_back(s);
        shift += SimpleSerializer::totSize(s);
        allNames += s + ", ";
    }

    if (allNames.length() >= 2) {
        allNames.pop_back();
        allNames.pop_back();
    }

    debug2("Plugin %s was set up to dump channels %s, path is %s", name.c_str(), allNames.c_str(), path.c_str());

    dumper = std::make_unique<XDMFParticlesDumper>(comm, nranks3D, path, channelNames, channelTypes);    
}

static void unpack_particles(const std::vector<Particle> &particles, std::vector<float> &pos, std::vector<float> &vel)
{
    int n = particles.size();
    pos.resize(3 * n);
    vel.resize(3 * n);

    for (int i = 0; i < n; ++i) {
        auto p = particles[i];
        pos[3*i + 0] = p.r.x;
        pos[3*i + 1] = p.r.y;
        pos[3*i + 2] = p.r.z;

        vel[3*i + 0] = p.u.x;
        vel[3*i + 1] = p.u.y;
        vel[3*i + 2] = p.u.z;
    }
}

void ParticleDumperPlugin::deserialize(MPI_Status& stat)
{
    float t;
    int totSize = 0;
    SimpleSerializer::deserialize(data, t, particles);
    totSize += SimpleSerializer::totSize(t, particles);

    debug2("Plugin %s will dump right now", name.c_str());

    int c = 0;
    for (auto& ch : channelData) {
        SimpleSerializer::deserialize(data.data() + totSize, ch);
        totSize += SimpleSerializer::totSize(ch);
        
        debug3("Received %d bytes, corresponding to the channel '%s'",
               SimpleSerializer::totSize(ch), channelNames[c].c_str());
        
        c++;
    }

    unpack_particles(particles, positions, velocities);
    
    std::vector<const float*> chPtrs;
    chPtrs.push_back((const float*) velocities.data());
    for (auto& ch : channelData)
        chPtrs.push_back((const float*)ch.data());

    dumper->dump(particles.size(), positions.data(), chPtrs, t);
}



