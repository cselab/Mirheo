#include "dumpavg.h"
#include "simple_serializer.h"
#include <sstream>

template<typename ValType>
__device__ ValType clear()
{ return 0; }

__device__ float clear<float>()
{ return 0.0f; }

__device__ float2 clear<float2>()
{ return make_float2(0.0f); }

__device__ float3 clear<float3>()
{ return make_float3(0.0f); }

__device__ float4 clear<float4>()
{ return make_float4(0.0f); }

__global__ void sample(const int * const __restrict__ cellsStart, const float4* __restrict__ coosvels, const float4* __restrict__ forces,
		const float mass, CellListInfo cinfo, float* avgDensity, float4* avgMomentum, float4* avgForce)
{
	const int cid = threadIdx.x + blockIdx.x*blockDim.x;

	if (cid < cinfo.totcells)
	{
		const int2 start_size = cinfo.decodeStartSize(cellsStart[cid]);

		if (avgDensity != nullptr)
			avgDensity[cid] += mass * start_size.y;

		// average!
		const float invnum = 1.0f / (float) start_size.y;
		for (int pid = start_size.x; pid < start_size.x + start_size.y; pid++)
		{
			if (avgMomentum != nullptr)
			{
				const float4 vel = coosvels[2*pid+1];
				avgMomentum[cid] += vel * (mass * invnum);
			}

			if (avgForce != nullptr)
			{
				const float4 frc = forces[pid];
				avgForce[cid] += frc * (mass * invnum);
			}
		}
	}
}

__global__ void scale(int n, float a, float4* res)
{
	const uint id = threadIdx.x + blockIdx.x*blockDim.x;

	if (id < n*4)
	{
		res[id] *= make_float4(a);
	}
}

Avg3DPlugin::Avg3DPlugin(int id, Simulation* sim, const MPI_Comm& comm, int sendRank,
			std::string pvNames, int sampleEvery, dumpEvery, int3 resolution, float3 h,
			bool needDensity, bool needMomentum, bool needForce) :
	SimulationPlugin(id, sim, stream, comm, sendRank),
	sampleEvery(sampleEvery), dumpEvery(dumpEvery), resolution(resolution), h(h),
	needDensity(needDensity), needMoment(needMoment), needForce(needForce),
	nTimeSteps(-1), nSamples(0)
{
	const int total = resolution.x * resolution.y * resolution.z;
	if (needDensity)  density .resize(total);
	if (needVelocity) velocity.resize(total);
	if (needForce)    force   .resize(total);

	std::stringstream sstream(pvNames);
	std::string name;
	std::vector<std::string> splitPvNames;

	while(std::getline(sstream, name, ','))
	{
		splitPvNames.push_back(name);
	}

	for (auto& nm : splitPvNames)
	{
		auto pvIter = sim->PVname2index.find(nm);
		if (pvIter == sim->PVname2index.end())
			die("No such particle vector registered: %s", nm.c_str());

		auto pv = sim->particleVectors[pvIter->second];
		auto cl = new CellList(pv, resolution, pv->domainStart, pv->length);
		particlesAndCells.push_back({pv, cl});
	}
}


void Avg3DPlugin::afterIntegration(float t)
{
	tm = t;
	nTimeSteps++;
	if (nTimeSteps % sampleEvery != 0) return;

	if (needDensity)  density. clear();
	if (needMomentum) momentum.clear();
	if (needForce)    force   .clear();

	for (auto pv_cl : particlesAndCells)
	{
		auto pv = pv_cl.first;
		auto cl = pv_cl.second;
		cl->build(stream);

		sample<<< (cl->totcells+127) / 128, 128 >>> (
				cl->cellsStart.constDevPtr(), pv->coosvels.constDevPtr(), pv->forces.constDevPtr(), pv->mass, cl->cellInfo(),
				needDensity  ? density .devPtr() : nullptr,
				needMomentum ? momentum.devPtr() : nullptr,
				needForce    ? force   .devPtr() : nullptr );
	}
	nSamples++;
}

void Avg3DPlugin::handshake()
{
	HostBuffer<char> data;
	SimpleSerializer::serialize(data, resolution, h, needDensity, needMomentum, needForce);
	MPI_Check( MPI_Send(data.constHostPtr(), data.size(), MPI_BYTE, sendRank, id, comm) );
}

void Avg3DPlugin::serializeAndSend()
{
	if (nTimeSteps % dumpEvery != 0) return;

	SimpleSerializer::serialize(sendBuffer, t, density, momentum, force);
	send(sendBuffer.constHostPtr(), sendBuffer.size());
}


Avg3DDumper::Avg3DDumper(int id, MPI_Comm comm, int recvRank, std::string path, std::vector<std::string> channelNames) :
		PostprocessPlugin(id, comm, recvRank)
{
}

void Avg3DDumper::handshake()
{
	HostBuffer<char> buf(1000);
	MPI_Check( MPI_Recv(buf.hostPtr(), buf.size(), MPI_BYTE, recvRank, id, comm) );
	SimpleSerializer::deserialize(buf, resolution, h, needDensity, needMomentum, needForce);

	std::vector<std::string> channelNames;

	if (needDensity)  channelNames.push_back("density");
	if (needMomentum) channelNames.push_back("momentum");
	if (needForce)    channelNames.push_back("force");

	dumper = new XDMFDumper(comm, path, dimensions, h, channelNames);
}

void Avg3DDumper::deserialize()
{
	float t;
	SimpleSerializer::deserialize(data, t, density, momentum, force);

	std::vector<float*> channels;
	if (needDensity)  channels.push_back(density.constHostPtr());
	if (needMomentum) channels.push_back(momentum.constHostPtr());
	if (needForce)    channels.push_back(force.constHostPtr());

	dumper->dump(channels, t);
}

