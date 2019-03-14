#include <mpi.h>

#include <core/xdmf/typeMap.h>
#include <core/utils/folders.h>
#include <core/xdmf/xdmf.h>

#include "particle_vector.h"
#include "restart_helpers.h"

LocalParticleVector::LocalParticleVector(ParticleVector* pv, int n) : pv(pv)
{
    resize_anew(n);
}

void LocalParticleVector::resize(const int n, cudaStream_t stream)
{
    if (n < 0) die("Tried to resize PV to %d < 0 particles", n);
    
    coosvels.        resize(n, stream);
    forces.          resize(n, stream);
    extraPerParticle.resize(n, stream);
    
    np = n;
}

void LocalParticleVector::resize_anew(const int n)
{
    if (n < 0) die("Tried to resize PV to %d < 0 particles", n);
    
    coosvels.        resize_anew(n);
    forces.          resize_anew(n);
    extraPerParticle.resize_anew(n);
    
    np = n;
}

LocalParticleVector::~LocalParticleVector() = default;


//============================================================================
// Particle Vector
//============================================================================

ParticleVector::ParticleVector(const YmrState *state, std::string name, float mass, int n) :
    ParticleVector(state, name, mass,
                   new LocalParticleVector(this, n),
                   new LocalParticleVector(this, 0) )
{}


std::vector<int> ParticleVector::getIndices_vector()
{
    auto& coosvels = local()->coosvels;
    coosvels.downloadFromDevice(0);
    
    std::vector<int> res(coosvels.size());
    for (int i = 0; i < coosvels.size(); i++)
        res[i] = coosvels[i].i1;
    
    return res;
}

PyTypes::VectorOfFloat3 ParticleVector::getCoordinates_vector()
{
    auto& coosvels = local()->coosvels;
    coosvels.downloadFromDevice(0);
    
    PyTypes::VectorOfFloat3 res(coosvels.size());
    for (int i = 0; i < coosvels.size(); i++)
    {
        float3 r = state->domain.local2global(coosvels[i].r);
        res[i] = { r.x, r.y, r.z };
    }
    
    return res;
}

PyTypes::VectorOfFloat3 ParticleVector::getVelocities_vector()
{
    auto& coosvels = local()->coosvels;
    coosvels.downloadFromDevice(0);
    
    PyTypes::VectorOfFloat3 res(coosvels.size());
    for (int i = 0; i < coosvels.size(); i++)
    {
        float3 u = coosvels[i].u;
        res[i] = { u.x, u.y, u.z };
    }
    
    return res;
}

PyTypes::VectorOfFloat3 ParticleVector::getForces_vector()
{
    HostBuffer<Force> forces;
    forces.copy(local()->forces, 0);
    
    PyTypes::VectorOfFloat3 res(forces.size());
    for (int i = 0; i < forces.size(); i++)
    {
        float3 f = forces[i].f;
        res[i] = { f.x, f.y, f.z };
    }
    
    return res;
}

void ParticleVector::setCoosVels_globally(PyTypes::VectorOfFloat6& coosvels, cudaStream_t stream)
{
    error("Not implemented yet");
/*    int c = 0;
    
    for (int i = 0; i < coosvels.size(); i++)
    {
        float3 r = { coosvels[i][0], coosvels[i][1], coosvels[i][2] };
        float3 u = { coosvels[i][3], coosvels[i][4], coosvels[i][5] };
        
        if (domain.inSubDomain(r))
        {
            c++;
            local()->resize(c, stream);
            
            local()->coosvels[c-1].r = domain.global2local( r );
            local()->coosvels[c-1].u = u;
        }
    }
    
    createIndicesHost();
    local()->coosvels.uploadToDevice(stream); */   
}

void ParticleVector::createIndicesHost()
{
    error("Not implemented yet");
//     int sz = local()->size();
//     for (int i=0; i<sz; i++)
//         local()->coosvels[i].i1 = i;
//     
//     int totalCount=0; // TODO: int64!
//     MPI_Check( MPI_Exscan(&sz, &totalCount, 1, MPI_INT, MPI_SUM, comm) );
//     
//     for (int i=0; i<sz; i++)
//         local()->coosvels[i].i1 += totalCount;
}

void ParticleVector::setCoordinates_vector(PyTypes::VectorOfFloat3& coordinates)
{
    auto& coosvels = local()->coosvels;
    
    if (coordinates.size() != local()->size())
        throw std::invalid_argument("Wrong number of particles passed, "
            "expected: " + std::to_string(local()->size()) +
            ", got: " + std::to_string(coordinates.size()) );
    
    for (int i = 0; i < coordinates.size(); i++)
    {
        auto& r = coordinates[i];
        coosvels[i].r = state->domain.global2local( float3{ r[0], r[1], r[2] } );
    }
    
    coosvels.uploadToDevice(0);
}

void ParticleVector::setVelocities_vector(PyTypes::VectorOfFloat3& velocities)
{
    auto& coosvels = local()->coosvels;
    
    if (velocities.size() != local()->size())
        throw std::invalid_argument("Wrong number of particles passed, "
        "expected: " + std::to_string(local()->size()) +
        ", got: " + std::to_string(velocities.size()) );
    
    for (int i = 0; i < velocities.size(); i++)
    {
        auto& u = velocities[i];
        coosvels[i].u = { u[0], u[1], u[2] };
    }
    
    coosvels.uploadToDevice(0);
}

void ParticleVector::setForces_vector(PyTypes::VectorOfFloat3& forces)
{
    HostBuffer<Force> myforces(local()->size());
    
    if (forces.size() != local()->size())
        throw std::invalid_argument("Wrong number of particles passed, "
        "expected: " + std::to_string(local()->size()) +
        ", got: " + std::to_string(forces.size()) );
    
    for (int i = 0; i < forces.size(); i++)
    {
        auto& f = forces[i];
        myforces[i].f = { f[0], f[1], f[2] };
    }
    
    local()->forces.copy(myforces, 0);
}


ParticleVector::~ParticleVector()
{ 
    delete _local;
    delete _halo;
    
}

ParticleVector::ParticleVector(const YmrState *state, std::string name,  float mass, LocalParticleVector *local, LocalParticleVector *halo) :
    YmrSimulationObject(state, name),
    mass(mass),
    _local(local),
    _halo(halo)
{
    // usually old positions and velocities don't need to exchanged
    requireDataPerParticle<Particle> (ChannelNames::oldParts, ExtraDataManager::PersistenceMode::None);
}

static void splitPV(DomainInfo domain, LocalParticleVector *local,
                    std::vector<float> &positions, std::vector<float> &velocities, std::vector<int> &ids)
{
    int n = local->size();
    positions.resize(3 * n);
    velocities.resize(3 * n);
    ids.resize(n);

    float3 *pos = (float3*) positions.data(), *vel = (float3*) velocities.data();
    
    for (int i = 0; i < n; i++)
    {
        auto p = local->coosvels[i];
        pos[i] = domain.local2global(p.r);
        vel[i] = p.u;
        ids[i] = p.i1;
    }
}

void ParticleVector::_extractPersistentExtraData(ExtraDataManager& extraData, std::vector<XDMF::Channel>& channels,
                                                 const std::set<std::string>& blackList)
{
    for (auto& namedChannelDesc : extraData.getSortedChannels())
    {        
        auto channelName = namedChannelDesc.first;
        auto channelDesc = namedChannelDesc.second;        
        
        if (channelDesc->persistence != ExtraDataManager::PersistenceMode::Persistent)
            continue;

        if (blackList.find(channelName) != blackList.end())
            continue;

        switch(channelDesc->dataType) {

#define SWITCH_ENTRY(ctype)                                             \
            case DataType::TOKENIZE(ctype):                             \
            {                                                           \
                auto buffer   = extraData.getData<ctype>(channelName);  \
                buffer->downloadFromDevice(0, ContainersSynch::Synch);  \
                auto type       = XDMF::getDataForm<ctype>();           \
                auto numbertype = XDMF::getNumberType<ctype>();         \
                auto datatype   = typeTokenize<ctype>();                \
                channels.push_back(XDMF::Channel(channelName, buffer->data(), type, numbertype, datatype )); \
            }                                                           \
            break;

            TYPE_TABLE(SWITCH_ENTRY);

#undef SWITCH_ENTRY
        };
    }
}

void ParticleVector::_extractPersistentExtraParticleData(std::vector<XDMF::Channel>& channels, const std::set<std::string>& blackList)
{
    auto& extraData = local()->extraPerParticle;
    _extractPersistentExtraData(extraData, channels, blackList);
}

void ParticleVector::_checkpointParticleData(MPI_Comm comm, std::string path)
{
    CUDA_Check( cudaDeviceSynchronize() );

    std::string filename = path + "/" + name + "-" + getStrZeroPadded(restartIdx);
    info("Checkpoint for particle vector '%s', writing to file %s", name.c_str(), filename.c_str());

    local()->coosvels.downloadFromDevice(0, ContainersSynch::Synch);

    auto positions = std::make_shared<std::vector<float>>();
    std::vector<float> velocities;
    std::vector<int> ids;
    splitPV(state->domain, local(), *positions, velocities, ids);

    XDMF::VertexGrid grid(positions, comm);

    std::vector<XDMF::Channel> channels;
    channels.push_back(XDMF::Channel("velocity", velocities.data(),
                                     XDMF::Channel::DataForm::Vector, XDMF::Channel::NumberType::Float, typeTokenize<float>() ));
    channels.push_back(XDMF::Channel(ChannelNames::globalIds, ids.data(),
                                     XDMF::Channel::DataForm::Scalar, XDMF::Channel::NumberType::Int, typeTokenize<int>() ));

    _extractPersistentExtraParticleData(channels);
    
    XDMF::write(filename, &grid, channels, comm);

    RestartHelpers::make_symlink(comm, path, name, filename);

    debug("Checkpoint for particle vector '%s' successfully written", name.c_str());
}

void ParticleVector::_getRestartExchangeMap(MPI_Comm comm, const std::vector<Particle> &parts, std::vector<int>& map)
{
    int dims[3], periods[3], coords[3];
    MPI_Check( MPI_Cart_get(comm, 3, dims, periods, coords) );

    map.resize(parts.size());
    
    for (int i = 0; i < parts.size(); ++i) {
        const auto& p = parts[i];
        int3 procId3 = make_int3(floorf(p.r / state->domain.localSize));

        if (procId3.x >= dims[0] || procId3.y >= dims[1] || procId3.z >= dims[2]) {
            map[i] = -1;
            continue;
        }
        
        int procId;
        MPI_Check( MPI_Cart_rank(comm, (int*)&procId3, &procId) );
        map[i] = procId;

        int rank;
        MPI_Comm_rank(comm, &rank);
    }
}

std::vector<int> ParticleVector::_restartParticleData(MPI_Comm comm, std::string path)
{
    CUDA_Check( cudaDeviceSynchronize() );

    std::string filename = path + "/" + name + ".xmf";
    info("Restarting particle vector %s from file %s", name.c_str(), filename.c_str());

    XDMF::readParticleData(filename, comm, this);

    std::vector<Particle> parts(local()->size());
    std::copy(local()->coosvels.begin(), local()->coosvels.end(), parts.begin());

    std::vector<int> map;
    
    _getRestartExchangeMap(comm, parts, map);
    RestartHelpers::exchangeData(comm, map, parts, 1);    
    RestartHelpers::copyShiftCoordinates(state->domain, parts, local());

    local()->coosvels.uploadToDevice(0);
    CUDA_Check( cudaDeviceSynchronize() );

    info("Successfully read %d particles", local()->coosvels.size());

    return map;
}

void ParticleVector::advanceRestartIdx()
{
    restartIdx = restartIdx xor 1;
}

void ParticleVector::checkpoint(MPI_Comm comm, std::string path)
{
    _checkpointParticleData(comm, path);
    advanceRestartIdx();
}

void ParticleVector::restart(MPI_Comm comm, std::string path)
{
    _restartParticleData(comm, path);
}


