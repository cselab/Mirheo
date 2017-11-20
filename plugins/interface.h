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

	SimulationPlugin(std::string name) :
		name(name), req(MPI_REQUEST_NULL)
	{}

	virtual void beforeForces     (cudaStream_t stream) {};
	virtual void beforeIntegration(cudaStream_t stream) {};
	virtual void afterIntegration (cudaStream_t stream) {};

	virtual void serializeAndSend (cudaStream_t stream) {};
	virtual void handshake() {};
	virtual void talk() {};

	virtual bool needPostproc() = 0;

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

	virtual ~SimulationPlugin() = default;

protected:
	Simulation* sim;
	MPI_Comm comm;
	MPI_Comm interComm;
	int rank;
	MPI_Request req;

	float currentTime;
	int currentTimeStep;

	std::hash<std::string> nameHash;
	int tag()
	{
		return (int)( nameHash(name) % 32767 );
	}

	void send(const std::vector<char>& data)
	{
		send(data.data(), data.size());
	}

	void send(const void* data, int sizeInBytes)
	{
		debug3("Plugin %s is sending now", name.c_str());
		MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );

		MPI_Check( MPI_Ssend(&sizeInBytes, 1, MPI_INT, rank, tag(), interComm) );
		MPI_Check( MPI_Issend(data, sizeInBytes, MPI_BYTE, rank, tag(), interComm, &req) );

		debug3("Plugin %s has sent the data (%d bytes)", name.c_str(), sizeInBytes);
	}
};

class PostprocessPlugin
{
public:
	std::string name;

	PostprocessPlugin(std::string name) : name(name) { }

	MPI_Request waitData()
	{
		MPI_Request req;
		MPI_Check( MPI_Irecv(&size, 1, MPI_INT, rank, tag(), interComm, &req) );
		return req;
	}

	void recv()
	{
		data.resize(size);
		MPI_Check( MPI_Recv(data.data(), size, MPI_BYTE, rank, tag(), interComm, MPI_STATUS_IGNORE) );

		debug3("Plugin %s has received the data (%d bytes)", name.c_str(), size);
	}

	virtual void deserialize(MPI_Status& stat) {};
	virtual void handshake() {};
	virtual void talk() {};

	virtual void setup(const MPI_Comm& comm, const MPI_Comm& interComm)
	{
		MPI_Check( MPI_Comm_dup(comm, &this->comm) );
		this->interComm = interComm;

		MPI_Check( MPI_Comm_rank(this->comm, &rank) );
	}

	virtual ~PostprocessPlugin() = default;


protected:
	MPI_Comm comm, interComm;
	int rank;
	std::vector<char> data;
	int size;

	std::hash<std::string> nameHash;
	int tag()
	{
		return (int)( nameHash(name) % 32767 );
	}
};





