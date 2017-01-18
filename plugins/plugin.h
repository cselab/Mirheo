#pragma once

#include <mpi.h>

class Simulation;

class SimulationPlugin
{
protected:
	Simulation* sim;
	const MPI_Comm& comm;
	int sendRank;
	MPI_Request req;
	cudaStream_t stream;
	float tm;

	int id;

	void send(const void* data, int size)
	{
		MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
		MPI_Check( MPI_Isend(data, size, MPI_BYTE, sendRank, id, comm, &req) );
	}

public:
	SimulationPlugin(const MPI_Comm& comm, int sendRank, int id = -1) :
		comm(comm), sendRank(sendRank), id(id), req(MPI_REQUEST_NULL) {};

	virtual void beforeForces(float t) {};
	virtual void beforeIntegration(float t) {};
	virtual void afterIntegration(float t) {};

	virtual void serializeAndSend() {};
	virtual void handshake() {};
	virtual void talk() {};

	void setup(Simulation* sim, cudaStream_t stream)
	{
		this->sim = sim;
		this->stream = stream;
	}

	virtual ~SimulationPlugin() {};
};

class PostprocessPlugin
{
protected:
	const MPI_Comm& comm;
	int recvRank;
	void *data;
	int size;

	int id;

public:
	PostprocessPlugin(const MPI_Comm& comm, int recvRank, int id = -1) :
			comm(comm), recvRank(recvRank), id(id) {};

	MPI_Request postRecv()
	{
		MPI_Request req;
		MPI_Check( MPI_Irecv(data, size, MPI_BYTE, recvRank, id, comm, &req) );
		return req;
	}

	virtual void deserialize() {};
	virtual void handshake() {};
	virtual void talk() {};

	virtual ~PostprocessPlugin() {};
};





