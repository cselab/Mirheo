#pragma once

#include "interface.h"

#include <core/containers.h>

class ParticleVector;
class CellList;

class SDFWall : public Wall
{
	friend void freezeParticlesInWall(Wall*, ParticleVector*, ParticleVector*, float, float);
	friend class MCMCSampler;

public:
	struct SdfInfo
	{
		cudaTextureObject_t sdfTex;
		float3 h, invh, extendedDomainSize;
		int3 resolution;
	};

private:
	MPI_Comm wallComm;

	PinnedBuffer<int> nBoundaryCells;
	std::vector<DeviceBuffer<int>> boundaryCells;

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

	SDFWall(std::string name, std::string sdfFileName, float3 sdfH);

	void setup(MPI_Comm& comm, float3 globalDomainSize, float3 globalDomainStart, float3 localDomainSize) override;

	void removeInner(ParticleVector* pv) override;
	void attach(ParticleVector* pv, CellList* cl) override;
	void bounce(float dt, cudaStream_t stream) override;

	void check(cudaStream_t stream) override;
};
