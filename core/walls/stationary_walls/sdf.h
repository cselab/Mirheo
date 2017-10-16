#pragma once

#include <core/containers.h>
#include <core/pvs/particle_vector.h>

class StationaryWall_SDF_Handler
{
public:
	__device__ __forceinline__ float operator()(const PVview view, float3 x) const
	{
		float3 texcoord = floorf((x + extendedDomainSize*0.5f) * invh);
		float3 lambda = (x - (texcoord * h - extendedDomainSize*0.5f)) * invh;

		const float s000 = tex3D<float>(sdfTex, texcoord.x + 0, texcoord.y + 0, texcoord.z + 0);
		const float s001 = tex3D<float>(sdfTex, texcoord.x + 1, texcoord.y + 0, texcoord.z + 0);
		const float s010 = tex3D<float>(sdfTex, texcoord.x + 0, texcoord.y + 1, texcoord.z + 0);
		const float s011 = tex3D<float>(sdfTex, texcoord.x + 1, texcoord.y + 1, texcoord.z + 0);
		const float s100 = tex3D<float>(sdfTex, texcoord.x + 0, texcoord.y + 0, texcoord.z + 1);
		const float s101 = tex3D<float>(sdfTex, texcoord.x + 1, texcoord.y + 0, texcoord.z + 1);
		const float s110 = tex3D<float>(sdfTex, texcoord.x + 0, texcoord.y + 1, texcoord.z + 1);
		const float s111 = tex3D<float>(sdfTex, texcoord.x + 1, texcoord.y + 1, texcoord.z + 1);

		const float s00x = s000 * (1 - lambda.x) + lambda.x * s001;
		const float s01x = s010 * (1 - lambda.x) + lambda.x * s011;
		const float s10x = s100 * (1 - lambda.x) + lambda.x * s101;
		const float s11x = s110 * (1 - lambda.x) + lambda.x * s111;

		const float s0yx = s00x * (1 - lambda.y) + lambda.y * s01x;
		const float s1yx = s10x * (1 - lambda.y) + lambda.y * s11x;

		const float szyx = s0yx * (1 - lambda.z) + lambda.z * s1yx;

		return szyx;
	}

protected:

	cudaTextureObject_t sdfTex;
	float3 h, invh, extendedDomainSize;
	int3 resolution;
};


class StationaryWall_SDF : public StationaryWall_SDF_Handler
{
public:
	void setup(MPI_Comm& comm, float3 globalDomainSize, float3 globalDomainStart, float3 localDomainSize);
	StationaryWall_SDF(std::string sdfFileName, float3 sdfH);


	StationaryWall_SDF(StationaryWall_SDF&&) = default;
	const StationaryWall_SDF_Handler& handler() const { return *(StationaryWall_SDF_Handler*)this; }

private:

	cudaArray *sdfArray;
	DeviceBuffer<float> sdfRawData; // TODO: this can be free'd after creation

	float3 sdfH;
	const float3 margin3{1, 1, 1};

	std::string sdfFileName;


	void readSdf(MPI_Comm& comm, int64_t fullSdfSize_byte, int64_t endHeader_byte, int nranks, int rank, std::vector<float>& fullSdfData);
	void readHeader(MPI_Comm& comm, int3& sdfResolution, float3& sdfExtent, int64_t& fullSdfSize_byte, int64_t& endHeader_byte, int rank);
	void prepareRelevantSdfPiece(int rank, const float* fullSdfData, float3 extendedDomainStart, float3 initialSdfH, int3 initialSdfResolution,
			int3& resolution, float3& offset, PinnedBuffer<float>& localSdfData);
};
