#pragma once

#include <core/containers.h>
#include <core/datatypes.h>
#include <core/logger.h>

#include <mpi.h>
#include <string>

class ParticleVector;
class CellList;

class Wall
{
	friend void freezeParticlesInWall(Wall*, ParticleVector*, ParticleVector*, float, float);
	friend class MCMCSampler;

public:
	std::string name;

	struct SdfInfo
	{
		cudaTextureObject_t sdfTex;
		float3 h, invh, extendedDomainSize;
		int3 resolution;
	};

private:
	MPI_Comm wallComm;

	std::vector<ParticleVector*> particleVectors;
	PinnedBuffer<int> nBoundaryCells;
	std::vector<DeviceBuffer<int>> boundaryCells;
	std::vector<CellList*> cellLists;

	SdfInfo sdfInfo;

	cudaArray *sdfArray;
	DeviceBuffer<float> sdfRawData; // TODO: this can be free'd after creation

	float3 sdfH;
	const float3 margin3{1, 1, 1};

	std::string sdfFileName;

	PinnedBuffer<int> nInside;

	void readSdf(int64_t fullSdfSize_byte, int64_t endHeader_byte, int nranks, int rank, std::vector<float>& fullSdfData);
	void readHeader(int3& sdfResolution, float3& sdfExtent, int64_t& fullSdfSize_byte, int64_t& endHeader_byte, int rank);
	void prepareRelevantSdfPiece(const float* fullSdfData, float3 extendedDomainStart, float3 initialSdfH, int3 initialSdfResolution,
			int3& resolution, float3& offset, PinnedBuffer<float>& localSdfData);

public:

	Wall(std::string name, std::string sdfFileName, float3 sdfH);

	void createSdf(MPI_Comm& comm, float3 globalDomainSize, float3 globalDomainStart, float3 localDomainSize);
	void removeInner(ParticleVector* pv);
	void attach(ParticleVector* pv, CellList* cl);
	void bounce(float dt, cudaStream_t stream);

	void check(cudaStream_t stream);
};
