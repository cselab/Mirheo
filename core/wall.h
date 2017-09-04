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

	ParticleVector* frozen;

	SdfInfo sdfInfo;

	cudaArray *sdfArray;
	DeviceBuffer<float> sdfRawData; // TODO: this can be free'd after creation

	float3 sdfH;
	float minSdf, maxSdf;

	const float3 margin3{1, 1, 1};

	std::string sdfFileName;

	// TODO:
	const float rc = 1.0f;

	void readSdf(int64_t fullSdfSize_byte, int64_t endHeader_byte, int nranks, int rank, std::vector<float>& fullSdfData);
	void readHeader(int3& sdfResolution, float3& sdfExtent, int64_t& fullSdfSize_byte, int64_t& endHeader_byte, int rank);
	void prepareRelevantSdfPiece(const float* fullSdfData, float3 extendedDomainStart, float3 initialSdfH, int3 initialSdfResolution,
			int3& resolution, float3& offset, PinnedBuffer<float>& localSdfData);

public:

	Wall(std::string name, std::string sdfFileName, float3 sdfH, float minSdf = 0.0f, float maxSdf = 1.2f);

	void createSdf(MPI_Comm& comm, float3 subDomainStart, float3 subDomaintSize, float3 globalDomainSize);
	void freezeParticles(ParticleVector* pv);
	void removeInner(ParticleVector* pv);
	void attach(ParticleVector* pv, CellList* cl);
	void bounce(float dt, cudaStream_t stream);

	void check(Particle* parts, int n, cudaStream_t stream);

	ParticleVector* getFrozen() { return frozen; }
};
