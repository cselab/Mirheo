#pragma once

#include <plugins/interface.h>
#include <core/containers.h>

#include <vector>

class ParticleVector;
class CellList;

class Average3D : public SimulationPlugin
{
public:
	enum class ChannelType : int
	{
		Scalar, Vector_float3, Vector_float4, Vector_2xfloat4, Tensor6
	};

	struct HostChannelsInfo
	{
		int n;
		std::vector<std::string> names;
		PinnedBuffer<ChannelType> types;
		PinnedBuffer<float*> averagePtrs, dataPtrs;
		std::vector<PinnedBuffer<float>> average;
	};

	Average3D(std::string name,
			  std::string pvName,
			  std::vector<std::string> channelNames, std::vector<Average3D::ChannelType> channelTypes,
			  int sampleEvery, int dumpEvery, float3 binSize);

	void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm) override;
	void handshake() override;
	void afterIntegration(cudaStream_t stream) override;
	void serializeAndSend(cudaStream_t stream) override;

	bool needPostproc() override { return true; }

	~Average3D() = default;

protected:
	std::string pvName;
	int nSamples;
	int sampleEvery, dumpEvery;
	int3 resolution;
	float3 binSize;

	PinnedBuffer<float>  density;
	std::vector<char> sendBuffer;

	ParticleVector* pv;

	HostChannelsInfo channelsInfo;

	void scaleSampled(cudaStream_t stream);
};

