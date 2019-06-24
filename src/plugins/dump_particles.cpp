#include "dump_particles.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/pvs/particle_vector.h>
#include <core/simulation.h>
#include <core/utils/folders.h>

ParticleSenderPlugin::ParticleSenderPlugin(const YmrState *state, std::string name, std::string pvName, int dumpEvery,
                                           std::vector<std::string> channelNames,
                                           std::vector<ChannelType> channelTypes) :
    SimulationPlugin(state, name),
    pvName(pvName),
    dumpEvery(dumpEvery),
    channelNames(channelNames),
    channelTypes(channelTypes)
{
    channelData.resize(channelNames.size());
}

void ParticleSenderPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

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

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, sizes, channelNames);
    send(sendBuffer);
}

void ParticleSenderPlugin::beforeForces(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    positions .genericCopy(&pv->local()->positions() , stream);
    velocities.genericCopy(&pv->local()->velocities(), stream);

    for (int i = 0; i < channelNames.size(); ++i) {
        auto name = channelNames[i];
        auto srcContainer = pv->local()->dataPerParticle.getGenericData(name);
        channelData[i].genericCopy(srcContainer, stream); 
    }
}

void ParticleSenderPlugin::serializeAndSend(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    debug2("Plugin %s is sending now data", name.c_str());
    
    for (auto& p : positions)
    {
        auto r = state->domain.local2global(make_float3(p));
        p.x = r.x; p.y = r.y; p.z = r.z;
    }

    YmrState::StepType timeStamp = getTimeStamp(state, dumpEvery);
    
    debug2("Plugin %s is packing now data consisting of %d particles", name.c_str(), positions.size());
    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, timeStamp, state->currentTime, positions, velocities, channelData);
    send(sendBuffer);
}




ParticleDumperPlugin::ParticleDumperPlugin(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(path),
    positions(std::make_shared<std::vector<float3>>())
{}

void ParticleDumperPlugin::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::vector<int> sizes;
    std::vector<std::string> names;
    SimpleSerializer::deserialize(data, sizes, names);
    
    auto init_channel = [] (XDMF::Channel::DataForm dataForm, int sz, const std::string& str,
                            XDMF::Channel::NumberType numberType = XDMF::Channel::NumberType::Float, TypeDescriptor datatype = DataTypeWrapper<float>()) {
        return XDMF::Channel(str, nullptr, dataForm, numberType, datatype);
    };

    // Velocity and id are special channels which are always present
    std::string allNames = "velocity, id";
    channels.push_back(init_channel(XDMF::Channel::DataForm::Vector, 3, "velocity", XDMF::Channel::NumberType::Float, DataTypeWrapper<float>()));
    channels.push_back(init_channel(XDMF::Channel::DataForm::Scalar, 1, "id", XDMF::Channel::NumberType::Int64, DataTypeWrapper<int64_t>()));

    for (int i = 0; i<sizes.size(); i++)
    {
        allNames += ", " + names[i];
        switch (sizes[i])
        {
            case 1: channels.push_back(init_channel(XDMF::Channel::DataForm::Scalar,  sizes[i], names[i])); break;
            case 3: channels.push_back(init_channel(XDMF::Channel::DataForm::Vector,  sizes[i], names[i])); break;
            case 6: channels.push_back(init_channel(XDMF::Channel::DataForm::Tensor6, sizes[i], names[i])); break;

            default:
                die("Plugin '%s' got %d as a channel '%s' size, expected 1, 3 or 6", name.c_str(), sizes[i], names[i].c_str());
        }
    }
    
    // Create the required folder
    createFoldersCollective(comm, parentPath(path));

    debug2("Plugin '%s' was set up to dump channels %s. Path is %s", name.c_str(), allNames.c_str(), path.c_str());
}

static void unpackParticles(const std::vector<float4> &pos4, const std::vector<float4> &vel4,
                            std::vector<float3> &pos, std::vector<float3> &vel, std::vector<int64_t> &ids)
{
    int n = pos4.size();
    pos.resize(n);
    vel.resize(n);
    ids.resize(n);

    for (int i = 0; i < n; ++i)
    {
        auto p = Particle(pos4[i], vel4[i]);
        pos[i] = p.r;
        vel[i] = p.u;
        ids[i] = p.getId();
    }
}

void ParticleDumperPlugin::_recvAndUnpack(YmrState::TimeType &time, YmrState::StepType& timeStamp)
{
    int c = 0;
    SimpleSerializer::deserialize(data, timeStamp, time, pos4, vel4, channelData);
        
    unpackParticles(pos4, vel4, *positions, velocities, ids);

    channels[c++].data = velocities.data();
    channels[c++].data = ids.data();
    
    for (int i = 0; i < channelData.size(); i++)
        channels[c++].data = channelData[i].data();
}

void ParticleDumperPlugin::deserialize(MPI_Status& stat)
{
    debug2("Plugin '%s' will dump right now", name.c_str());

    YmrState::TimeType time;
    YmrState::StepType timeStamp;
    _recvAndUnpack(time, timeStamp);
    
    std::string fname = path + getStrZeroPadded(timeStamp, zeroPadding);
    
    XDMF::VertexGrid grid(positions, comm);
    XDMF::write(fname, &grid, channels, time, comm);
}



