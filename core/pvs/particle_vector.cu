#include <mpi.h>

#include "particle_vector.h"

void ParticleVector::checkpoint(MPI_Comm comm, std::string path)
{
	CUDA_Check( cudaDeviceSynchronize() );

	std::string fname = path + "/" + name + ".chk";
	info("Checkpoint for particle vector %s, writing file %s", name.c_str(), fname.c_str());

	local()->coosvels.downloadFromDevice(0, true);
	for (int i=0; i<local()->coosvels.size(); i++)
		local()->coosvels[i].r = local2global(local()->coosvels[i].r);

	int myrank, size;
	MPI_Check( MPI_Comm_rank(comm, &myrank) );
	MPI_Check( MPI_Comm_size(comm, &size) );

	MPI_Datatype ptype;
	MPI_Check( MPI_Type_contiguous(sizeof(Particle), MPI_CHAR, &ptype) );
	MPI_Check( MPI_Type_commit(&ptype) );

	int64_t mysize = local()->coosvels.size();
	int64_t offset = 0;
	MPI_Check( MPI_Exscan(&mysize, &offset, 1, MPI_LONG_LONG, MPI_SUM, comm) );

	MPI_File f;
	MPI_Status status;

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
		int3 procId3 = make_int3(floorf(p.r / localDomainSize));

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
		MPI_Check( MPI_Recv(addr, msize, ptype, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE) );
	}

	for (int i=0; i<local()->coosvels.size(); i++)
		local()->coosvels[i].r = global2local(local()->coosvels[i].r);

	local()->coosvels.uploadToDevice(0);

	CUDA_Check( cudaDeviceSynchronize() );

	info("Successfully grabbed %d particles out of total %lld", local()->coosvels.size(), total);

	MPI_Check( MPI_Type_free(&ptype) );
}














