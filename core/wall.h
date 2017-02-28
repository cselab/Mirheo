#pragma once

#include <core/datatypes.h>
#include <core/containers.h>
#include <core/logger.h>
#include <core/celllist.h>

#include <mpi.h>
#include <string>

class Wall
{
public:
	std::string name;

private:
	MPI_Comm wallComm;

	std::vector<ParticleVector*> particleVectors;
	PinnedBuffer<int> nBoundaryCells;
	std::vector<DeviceBuffer<int>> boundaryCells;
	std::vector<CellList*> cellLists;

	ParticleVector frozen;

	cudaTextureObject_t sdfTex;
	cudaArray *sdfArray;
	DeviceBuffer<float> sdfRawData; // TODO: this can be free'd after creation

	float3 sdfH;
	float3 subDomainSize, globalDomainSize;
	int3 localSdfResolution, globalResolution;

	std::string sdfFileName;
	float _creationTime;

	// TODO:
	const float rc = 1.0f;

	void readSdf(int64_t fullSdfSize_byte, int64_t endHeader_byte, int nranks, int rank, std::vector<float>& fullSdfData);
	void readHeader(int3& sdfResolution, float3& sdfExtent, int64_t& fullSdfSize_byte, int64_t& endHeader_byte, int rank);

public:

	Wall(std::string name, std::string sdfFileName, float3 sdfH, float _creationTime);

	void create(MPI_Comm& comm, float3 subDomainStart, float3 subDomaintSize, float3 globalDomainSize, ParticleVector* pv, CellList* cl);
	void attach(ParticleVector* pv, CellList* cl);
	void bounce(float dt, cudaStream_t stream);

	ParticleVector* getFrozen() { return &frozen; }
	float creationTime() { return _creationTime; }
};
