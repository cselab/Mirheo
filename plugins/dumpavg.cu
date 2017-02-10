#include "dumpavg.h"
#include "simple_serializer.h"
#include "../core/simulation.h"
#include "../core/containers.h"
#include "../core/celllist.h"
#include "../core/helper_math.h"
#include <sstream>

__global__ void sample(const int * const __restrict__ cellsStart, const float4* __restrict__ coosvels, const float4* __restrict__ forces,
		const float mass, CellListInfo cinfo, float* avgDensity, float3* avgMomentum, float3* avgForce)
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
				avgMomentum[cid] += make_float3(vel * (mass * invnum));
			}

			if (avgForce != nullptr)
			{
				const float4 frc = forces[pid];
				avgForce[cid] += make_float3(frc * (mass * invnum));
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

Avg3DPlugin::Avg3DPlugin(std::string name, std::string pvNames, int sampleEvery, int dumpEvery, int3 resolution, float3 h,
			bool needDensity, bool needMomentum, bool needForce) :
	SimulationPlugin(name),
	sampleEvery(sampleEvery), dumpEvery(dumpEvery), resolution(resolution), h(h),
	needDensity(needDensity), needMomentum(needMomentum), needForce(needForce),
	nTimeSteps(-1), nSamples(0)
{
	const int total = resolution.x * resolution.y * resolution.z;
	if (needDensity)  density .resize(total);
	if (needMomentum) momentum.resize(total);
	if (needForce)    force   .resize(total);

	std::stringstream sstream(pvNames);
	std::string pvName;
	std::vector<std::string> splitPvNames;

	while(std::getline(sstream, pvName, ','))
	{
		splitPvNames.push_back(pvName);
	}

	for (auto& nm : splitPvNames)
	{
		auto pvMap = sim->getPvMap();
		auto pvIter = pvMap.find(nm);
		if (pvIter == pvMap.end())
			die("No such particle vector registered: %s", nm.c_str());

		auto pv = sim->getParticleVectors()[pvIter->second];
		auto cl = new CellList(pv, resolution, pv->domainStart, pv->domainLength);
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
				(int*)cl->cellsStart.devPtr(), (float4*)pv->coosvels.devPtr(), (float4*)pv->forces.devPtr(),
				pv->mass, cl->cellInfo(),
				needDensity  ? density .devPtr() : nullptr,
				needMomentum ? momentum.devPtr() : nullptr,
				needForce    ? force   .devPtr() : nullptr );
	}
	nSamples++;
}

void Avg3DPlugin::serializeAndSend()
{
	if (nTimeSteps % dumpEvery != 0) return;

	if (needDensity)  density .downloadFromDevice();
	if (needMomentum) momentum.downloadFromDevice();
	if (needForce)    force   .downloadFromDevice();

	SimpleSerializer::serialize(sendBuffer, tm, density, momentum, force);
	send(sendBuffer.hostPtr(), sendBuffer.size());
}

void Avg3DPlugin::handshake()
{
	HostBuffer<char> data;
	SimpleSerializer::serialize(data, resolution, h, needDensity, needMomentum, needForce);
	MPI_Check( MPI_Send(data.hostPtr(), data.size(), MPI_BYTE, rank, id, interComm) );
}





Avg3DDumper::Avg3DDumper(std::string name, std::string path, int3 nranks3D) :
		PostprocessPlugin(name), path(path), nranks3D(nranks3D) { }

void Avg3DDumper::handshake()
{
	HostBuffer<char> buf(1000);
	MPI_Check( MPI_Recv(buf.hostPtr(), buf.size(), MPI_BYTE, rank, id, interComm, MPI_STATUS_IGNORE) );
	SimpleSerializer::deserialize(buf, resolution, h, needDensity, needMomentum, needForce);

	std::vector<std::string> channelNames;
	std::vector<XDMFDumper::ChannelType> channelTypes;

	if (needDensity)
	{
		channelNames.push_back("density");
		channelTypes.push_back(XDMFDumper::ChannelType::Scalar);
	}
	if (needMomentum)
	{
		channelNames.push_back("momentum");
		channelTypes.push_back(XDMFDumper::ChannelType::Vector);
	}
	if (needForce)
	{
		channelNames.push_back("force");
		channelTypes.push_back(XDMFDumper::ChannelType::Vector);
	}


	dumper = new XDMFDumper(comm, nranks3D, path, resolution, h, channelNames, channelTypes);
}

void Avg3DDumper::deserialize(MPI_Status& stat)
{
	float t;
	SimpleSerializer::deserialize(data, t, density, momentum, force);

	std::vector<const float*> channels;
	if (needDensity)  channels.push_back(density.hostPtr());
	if (needMomentum) channels.push_back((const float*)momentum.hostPtr());
	if (needForce)    channels.push_back((const float*)force.hostPtr());

	dumper->dump(channels, t);
}

