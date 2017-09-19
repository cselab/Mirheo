#pragma once

#include <mpi.h>
#include <string>
#include <core/logger.h>
#include <core/datatypes.h>

class Simulation;

// TODO: variable size messages

class SimulationPlugin
{
public:
	std::string name;

	bool requirePostproc;

protected:
	Simulation* sim;
	MPI_Comm comm;
	MPI_Comm interComm;
	int rank;
	MPI_Request req;

	float currentTime;
	int currentTimeStep;

	int id;

	void send(const void* data, int sizeInBytes)
	{
		debug3("Plugin %s is sending now", name.c_str());
		MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
		MPI_Check( MPI_Isend(data, sizeInBytes, MPI_BYTE, rank, id, interComm, &req) );
		debug3("Plugin %s has sent the data", name.c_str());
	}

public:
	SimulationPlugin(std::string name, bool requirePostproc = false) :
		name(name), req(MPI_REQUEST_NULL), requirePostproc(requirePostproc)
	{}

	virtual void beforeForces     (cudaStream_t stream) {};
	virtual void beforeIntegration(cudaStream_t stream) {};
	virtual void afterIntegration (cudaStream_t stream) {};

	virtual void serializeAndSend (cudaStream_t stream) {};
	virtual void handshake() {};
	virtual void talk() {};

	void setTime(float t, int tstep)
	{
		currentTime = t;
		currentTimeStep = tstep;
	}

	virtual void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
	{
		this->sim = sim;

		MPI_Check( MPI_Comm_dup(comm, &this->comm) );
		this->interComm = interComm;

		MPI_Check( MPI_Comm_rank(this->comm, &rank) );
	}
	void setId(int id) { this->id = id; }

	virtual ~SimulationPlugin() = default;
};

class PostprocessPlugin
{
public:
	std::string name;

protected:
	MPI_Comm comm, interComm;
	int rank;
	std::vector<char> data;
	int size;

	int id;

public:
	PostprocessPlugin(std::string name) : name(name) { }

	MPI_Request postRecv()
	{
		MPI_Request req;
		MPI_Check( MPI_Irecv(data.data(), size, MPI_BYTE, rank, id, interComm, &req) );
		return req;
	}

	virtual void deserialize(MPI_Status& stat) {};
	virtual void handshake() {};
	virtual void talk() {};

	void setup(const MPI_Comm& comm, const MPI_Comm& interComm)
	{
		MPI_Check( MPI_Comm_dup(comm, &this->comm) );
		this->interComm = interComm;

		MPI_Check( MPI_Comm_rank(this->comm, &rank) );
	}
	void setId(int id) { this->id = id; }

	virtual ~PostprocessPlugin() = default;
};





