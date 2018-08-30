#include <mpi.h>

#include "particle_vector.h"

// Local coordinate system; (0,0,0) is center of the local domain
LocalParticleVector::LocalParticleVector(ParticleVector* pv, int n) : pv(pv)
{
    resize_anew(n);
}

void LocalParticleVector::resize(const int n, cudaStream_t stream)
{
    if (n < 0) die("Tried to resize PV to %d < 0 particles", n);
    
    //debug4("Resizing PV '%s' (%s) from %d particles to %d particles",
    //        pv->name.c_str(), this == pv->local() ? "local" : "halo",
    //        np, n);
    
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

ParticleVector::ParticleVector(std::string name, float mass, int n) :
ParticleVector(
    name, mass,
    new LocalParticleVector(this, n),
                new LocalParticleVector(this, 0) )
{    }


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
        float3 r = domain.local2global(coosvels[i].r);
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
        coosvels[i].r = domain.global2local( float3{ r[0], r[1], r[2] } );
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

ParticleVector::ParticleVector(    std::string name, float mass, LocalParticleVector *local, LocalParticleVector *halo ) :
name(name), mass(mass), _local(local), _halo(halo)
{
    // usually old positions and velocities don't need to exchanged
    requireDataPerParticle<Particle> ("old_particles", false);
}
    

void ParticleVector::checkpoint(MPI_Comm comm, std::string path)
{
    CUDA_Check( cudaDeviceSynchronize() );

    std::string fname = path + "/" + name + std::to_string(restartIdx) + ".chk";
    info("Checkpoint for particle vector '%s', writing to file %s", name.c_str(), fname.c_str());

    local()->coosvels.downloadFromDevice(0, ContainersSynch::Synch);

    for (int i=0; i<local()->size(); i++)
        local()->coosvels[i].r = domain.local2global(local()->coosvels[i].r);

    int myrank, size;
    MPI_Check( MPI_Comm_rank(comm, &myrank) );
    MPI_Check( MPI_Comm_size(comm, &size) );

    MPI_Datatype ptype;
    MPI_Check( MPI_Type_contiguous(sizeof(Particle), MPI_CHAR, &ptype) );
    MPI_Check( MPI_Type_commit(&ptype) );

    int64_t mysize = local()->size();
    int64_t offset = 0;
    MPI_Check( MPI_Exscan(&mysize, &offset, 1, MPI_LONG_LONG, MPI_SUM, comm) );

    MPI_File f;
    MPI_Status status;

    // Remove previous file if it was there
    MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE|MPI_MODE_DELETE_ON_CLOSE|MPI_MODE_WRONLY, MPI_INFO_NULL, &f) );
    MPI_Check( MPI_File_close(&f) );

    // Open for real now
    MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f) );
    if (myrank == size-1)
    {
        const int64_t total = offset + mysize;
        MPI_Check( MPI_File_write_at(f, 0, &total, 1, MPI_LONG_LONG, &status) );
    }

    const int64_t header = (sizeof(int64_t) + sizeof(Particle) - 1) / sizeof(Particle);
    MPI_Check( MPI_File_write_at_all(f, (offset + header)*sizeof(Particle), local()->coosvels.hostPtr(), (int)mysize, ptype, &status) );
    MPI_Check( MPI_File_close(&f) );

    MPI_Check( MPI_Type_free(&ptype) );

    if (myrank == 0)
    {
        std::string lnname = path + "/" + name + ".chk";

        std::string command = "ln -f " + fname + "  " + lnname;
        if ( system(command.c_str()) != 0 )
            error("Could not create link for checkpoint file of PV '%s'", name.c_str());
    }

    debug("Checkpoint for particle vector '%s' successfully written", name.c_str());
    restartIdx = restartIdx xor 1;
}

void ParticleVector::restart(MPI_Comm comm, std::string path)
{
    CUDA_Check( cudaDeviceSynchronize() );

    std::string fname = path + "/" + name + ".chk";
    info("Restarting particle vector %s from file %s", name.c_str(), fname.c_str());

    int myrank, commSize;
    int dims[3], periods[3], coords[3];
    MPI_Check( MPI_Comm_rank(comm, &myrank) );
    MPI_Check( MPI_Comm_size(comm, &commSize) );
    MPI_Check( MPI_Cart_get(comm, 3, dims, periods, coords) );

    MPI_Datatype ptype;
    MPI_Check( MPI_Type_contiguous(sizeof(Particle), MPI_CHAR, &ptype) );
    MPI_Check( MPI_Type_commit(&ptype) );

    // Find size of data chunk to read
    MPI_File f;
    MPI_Status status;
    int64_t total;
    MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &f) );
    if (myrank == 0)
        MPI_Check( MPI_File_read_at(f, 0, &total, 1, MPI_LONG_LONG, &status) );
    MPI_Check( MPI_Bcast(&total, 1, MPI_LONG_LONG, 0, comm) );

    int64_t sizePerProc = (total+commSize-1) / commSize;
    int64_t offset = sizePerProc * myrank;
    int64_t mysize = std::min(offset+sizePerProc, total) - offset;

    debug2("Will read %lld particles from the file", mysize);

    // Read your chunk
    std::vector<Particle> readBuf(mysize);
    const int64_t header = (sizeof(int64_t) + sizeof(Particle) - 1) / sizeof(Particle);
    MPI_Check( MPI_File_read_at_all(f, (offset + header)*sizeof(Particle), readBuf.data(), mysize, ptype, &status) );
    MPI_Check( MPI_File_close(&f) );

    // Find where to send the read particles
    std::vector<std::vector<Particle>> sendBufs(commSize);
    for (auto& p : readBuf)
    {
        int3 procId3 = make_int3(floorf(p.r / domain.localSize));

        if (procId3.x >= dims[0] || procId3.y >= dims[1] || procId3.z >= dims[2])
            continue;

        int procId;
        MPI_Check( MPI_Cart_rank(comm, (int*)&procId3, &procId) );
        sendBufs[procId].push_back(p);
    }

    // Do the send
    std::vector<MPI_Request> reqs(commSize);
    for (int i=0; i<commSize; i++)
    {
        debug3("Sending %d paricles to rank %d", sendBufs[i].size(), i);
        MPI_Check( MPI_Isend(sendBufs[i].data(), sendBufs[i].size(), ptype, i, 0, comm, reqs.data()+i) );
    }

    int curSize = 0;
    local()->resize(curSize, 0);
    for (int i=0; i<commSize; i++)
    {
        MPI_Status status;
        int msize;
        MPI_Check( MPI_Probe(MPI_ANY_SOURCE, 0, comm, &status) );
        MPI_Check( MPI_Get_count(&status, ptype, &msize) );

        local()->resize(curSize + msize, 0);
        Particle* addr = local()->coosvels.hostPtr() + curSize;
        curSize += msize;

        debug3("Receiving %d particles from ???", msize);
        MPI_Check( MPI_Recv(addr, msize, ptype, status.MPI_SOURCE, 0, comm, MPI_STATUS_IGNORE) );
    }

    for (int i=0; i<local()->coosvels.size(); i++)
        local()->coosvels[i].r = domain.global2local(local()->coosvels[i].r);

    local()->coosvels.uploadToDevice(0);

    CUDA_Check( cudaDeviceSynchronize() );

    info("Successfully grabbed %d particles out of total %lld", local()->coosvels.size(), total);

    MPI_Check( MPI_Waitall(commSize, reqs.data(), MPI_STATUSES_IGNORE) );
    MPI_Check( MPI_Type_free(&ptype) );
}














