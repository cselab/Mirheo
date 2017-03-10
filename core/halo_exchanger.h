#pragma once

#include <core/datatypes.h>
#include <core/logger.h>

#include <vector>
//#include <thread>

class ObjectVector;
class ParticleVector;
class CellList;

struct HaloHelper
{
	PinnedBuffer<int> counts;
	PinnedBuffer<char>  sendBufs[27];
	PinnedBuffer<char*> sendAddrs;
	PinnedBuffer<char>  recvBufs[27];
	PinnedBuffer<char*> recvAddrs;

	std::vector<int> recvOffsets;
	std::vector<MPI_Request> requests;

	cudaStream_t stream;
	//std::thread thread;

	HaloHelper(ParticleVector* pv, const int sizes[3]);
};

class HaloExchanger
{
private:
	int dir2rank[27];
	int compactedDirs[27]; // nActiveNeighbours entries s.t. we need to send/recv to/from dir2rank[compactedDirs[i]], i=0..nActiveNeighbours

	int nActiveNeighbours;
	int myrank;
	MPI_Datatype mpiPartType, mpiForceType;
	MPI_Comm haloComm;

	cudaStream_t defStream;

	std::vector<std::pair<ParticleVector*, CellList*>> particlesAndCells;
	std::vector<HaloHelper*> helpers;

	std::vector<std::pair<ObjectVector*, float>> objectsAndRCs;
	std::vector<HaloHelper*> objectHelpers;

	void exchange(std::string pvName, HaloHelper* helper, int typeSize);
	void uploadHalos(ParticleVector* pv, HaloHelper* helper);
	void prepareForces(ObjectVector* ov, HaloHelper* helper);
	void uploadForces (ObjectVector* ov, HaloHelper* helper);

public:

	void _prepareHalos(ParticleVector* pv, CellList* cl, HaloHelper* helper);
	void _prepareObjectHalos(ObjectVector* ov, float rc, HaloHelper* helper);

	HaloExchanger(MPI_Comm& comm, cudaStream_t defStream);
	void attach(ParticleVector* pv, CellList* cl);
	void attach(ObjectVector* ov, float rc);
	void init();
	void finalize();
	void exchangeForces();
};
