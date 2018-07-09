#include "object_vector.h"
#include "views/ov.h"

#include <core/utils/kernel_launch.h>
#include <core/utils/cuda_common.h>

__global__ void min_max_com(OVview ovView)
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

void ObjectVector::findExtentAndCOM(cudaStream_t stream, bool isLocal)
{
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
            min_max_com,
            (ovView.nObjects*32 + nthreads-1)/nthreads, nthreads, 0, stream,
            ovView );
}


void ObjectVector::restart(MPI_Comm comm, std::string path)
{
    CUDA_Check( cudaDeviceSynchronize() );

    std::string fname = path + "/" + name + ".chk";
    info("Restarting object vector %s from file %s", name.c_str(), fname.c_str());

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

    // Read only full objects
    int64_t totObjs = total / objSize;
    if (totObjs * objSize != total)
        die("Restart file for OV '%s' is probably corrupt, number of particles %d is not divisible by object size %d",
                name.c_str(), total, objSize);

    int64_t objsPerProc = (totObjs+commSize-1) / commSize;
    int64_t objOffset = objsPerProc * myrank;
    int64_t myObjsize = std::min(objOffset+objsPerProc, totObjs) - objOffset;

    int64_t offset = objOffset * objSize;
    int64_t mysize = myObjsize * objSize;

    debug2("Will read %lld particles from the file", mysize);

    // Read your chunk
    std::vector<Particle> readBuf(mysize);
    const int64_t header = (sizeof(int64_t) + sizeof(Particle) - 1) / sizeof(Particle);
    MPI_Check( MPI_File_read_at_all(f, (offset + header)*sizeof(Particle), readBuf.data(), mysize, ptype, &status) );
    MPI_Check( MPI_File_close(&f) );

    // Find where to send the read particles
    std::vector<std::vector<Particle>> sendBufs(commSize);
    for (int objId = 0; objId < myObjsize; objId++)
    {
        float3 com = make_float3(0);
        for (int i = 0; i < objSize; i++)
        {
            Particle p = readBuf[objId*objSize + i];
            com += p.r;
        }

        com /= objSize;

        int3 procId3 = make_int3(floorf(com / domain.localSize));

        if (procId3.x >= dims[0] || procId3.y >= dims[1] || procId3.z >= dims[2])
            continue;

        int procId;
        MPI_Check( MPI_Cart_rank(comm, (int*)&procId3, &procId) );
        sendBufs[procId].insert( sendBufs[procId].end(), readBuf.begin() + objId*objSize, readBuf.begin() + (objId+1)*objSize );
    }

    // Same as for the particle vector
    // Maybe move to a separate function and reuse

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

        debug3("Receiving %d particles from %d", msize, status.MPI_SOURCE);
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
