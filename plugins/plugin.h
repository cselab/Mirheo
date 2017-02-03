#pragma once

#include <mpi.h>
#include "../core/logger.h"
#include "../core/datatypes.h"

class Simulation;

class SimulationPlugin
{
protected:
	Simulation* sim;
	MPI_Comm comm;
	MPI_Comm interComm;
	int rank;
	MPI_Request req;
	cudaStream_t stream;
	float tm;

	int id;

	void send(const void* data, int size)
	{
		MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
		MPI_Check( MPI_Isend(data, size, MPI_BYTE, rank, id, interComm, &req) );
	}

public:
	SimulationPlugin() : req(MPI_REQUEST_NULL) {};

	virtual void beforeForces(float t) {};
	virtual void beforeIntegration(float t) {};
	virtual void afterIntegration(float t) {};

	virtual void serializeAndSend() {};
	virtual void handshake() {};
	virtual void talk() {};

	void setup(Simulation* sim, cudaStream_t stream, MPI_Comm& comm, MPI_Comm& interComm)
	{
		this->sim       = sim;
		this->stream    = stream;

		MPI_Check( MPI_Comm_dup(comm,      &this->comm) );
		MPI_Check( MPI_Comm_dup(interComm, &this->interComm) );

		MPI_Check( MPI_Comm_rank(comm, &rank) );
	}
	void setId(int id) { this->id = id; }

	virtual ~SimulationPlugin() {};
};

class PostprocessPlugin
{
protected:
	MPI_Comm comm, interComm;
	int rank;
	HostBuffer<char> data;
	int size;

	int id;

public:
	MPI_Request postRecv()
	{
		MPI_Request req;
		MPI_Check( MPI_Irecv(data.hostPtr(), size, MPI_BYTE, rank, id, interComm, &req) );
		return req;
	}

	virtual void deserialize() {};
	virtual void handshake() {};
	virtual void talk() {};

	void setup(MPI_Comm& comm, MPI_Comm& interComm)
	{
		MPI_Check( MPI_Comm_dup(comm,      &this->comm) );
		MPI_Check( MPI_Comm_dup(interComm, &this->interComm) );

		MPI_Check( MPI_Comm_rank(this->comm, &rank) );
	}
	void setId(int id) { this->id = id; }

	virtual ~PostprocessPlugin() {};
};





