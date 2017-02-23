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

		float3 mymomentum{0.0f, 0.0f, 0.0f};
		float3 myforce   {0.0f, 0.0f, 0.0f};
		for (int pid = start_size.x; pid < start_size.x + start_size.y; pid++)
		{
			if (avgMomentum != nullptr)
			{
				const float4 vel = coosvels[2*pid+1];
				mymomentum += make_float3(vel * (mass * invnum));
			}

			if (avgForce != nullptr)
			{
				const float4 frc = forces[pid];
				myforce += make_float3(frc * invnum);
			}
		}

		if (avgMomentum != nullptr) avgMomentum[cid] += mymomentum;
		if (avgForce    != nullptr) avgForce   [cid] += myforce;
	}
}

__global__ void scale(int n, float a, float* res)
{
	const int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < n) res[id] *= a;
}

Avg3DPlugin::Avg3DPlugin(std::string name, std::string pvNames, int sampleEvery, int dumpEvery, int3 resolution,
			bool needDensity, bool needMomentum, bool needForce) :
	SimulationPlugin(name), pvNames(pvNames),
	sampleEvery(sampleEvery), dumpEvery(dumpEvery), resolution(resolution),
	needDensity(needDensity), needMomentum(needMomentum), needForce(needForce),
	nSamples(0)
{
	// TODO: this should be reworked if the domains are allowed to have different size

	const int total = resolution.x * resolution.y * resolution.z;
	if (needDensity)  density .resize(total);
	if (needMomentum) momentum.resize(total);
	if (needForce)    force   .resize(total);
}

void Avg3DPlugin::setup(Simulation* sim, cudaStream_t stream, const MPI_Comm& comm, const MPI_Comm& interComm)
{
	SimulationPlugin::setup(sim, stream, comm, interComm);

	std::stringstream sstream(pvNames);
	std::string pvName;
	std::vector<std::string> splitPvNames;

	while(std::getline(sstream, pvName, ','))
	{
		splitPvNames.push_back(pvName);
	}

	h =  sim->subDomainSize / make_float3(resolution);

	density.pushStream(stream);
	density.clearDevice();

	momentum.pushStream(stream);
	momentum.clearDevice();

	force.pushStream(stream);
	force.clearDevice();

	for (auto& nm : splitPvNames)
	{
		auto& pvMap = sim->getPvMap();
		auto pvIter = pvMap.find(nm);
		if (pvIter == pvMap.end())
			die("No such particle vector registered: %s", nm.c_str());

		auto pv = sim->getParticleVectors()[pvIter->second];
		auto cl = new CellList(pv, resolution, pv->domainStart, pv->domainLength);
		cl->setStream(stream);
		particlesAndCells.push_back({pv, cl});
	}

	info("Plugin %s was set up for the following particle vectors: %s", name.c_str(), pvNames.c_str());
}



void Avg3DPlugin::afterIntegration(bool& reordered)
{
	reordered = false;
	if (currentTimeStep % sampleEvery != 0 || currentTimeStep == 0) return;

	debug2("Plugin %s is sampling now", name.c_str());
	//reordered = true;

	for (auto pv_cl : particlesAndCells)
	{
		auto pv = pv_cl.first;
		auto cl = pv_cl.second;
		cl->build(stream);

		sample<<< (cl->totcells+127) / 128, 128, 0, stream >>> (
				(int*)cl->cellsStart.devPtr(), (float4*)pv->coosvels.devPtr(), (float4*)pv->forces.devPtr(),
				pv->mass, cl->cellInfo(),
				needDensity  ? density .devPtr() : nullptr,
				needMomentum ? momentum.devPtr() : nullptr,
				needForce    ? force   .devPtr() : nullptr );
	}

	// Roll back the reordering
	CUDA_Check( cudaStreamSynchronize(stream) );
	for (auto pv_cl : particlesAndCells)
	{
		auto pv = pv_cl.first;
		containerSwap(pv->coosvels, pv->pingPongBuf);
	}

	nSamples++;
}

void Avg3DPlugin::serializeAndSend()
{
	if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

	if (needDensity)
	{
		int sz = density.size();
		scale<<< (sz+127)/128, 128, 0, stream >>> ( sz, 1.0/nSamples, (float*)density.devPtr() );
		density.downloadFromDevice();
		density.clearDevice();
	}

	if (needMomentum)
	{
		int sz = momentum.size()*3;
		scale<<< (sz+127)/128, 128, 0, stream >>> ( sz, 1.0/nSamples, (float*)momentum.devPtr() );
		momentum.downloadFromDevice();
		momentum.clearDevice();
	}

	if (needForce)
	{
		int sz = force.size()*3;
		scale<<< (sz+127)/128, 128, 0, stream >>> ( sz, 1.0/nSamples, (float*)force.devPtr() );
		force.downloadFromDevice();
		force.clearDevice();
	}

	debug2("Plugin %s is sending now data", name.c_str());
	SimpleSerializer::serialize(sendBuffer, currentTime, density, momentum, force);
	send(sendBuffer.hostPtr(), sendBuffer.size());

	nSamples = 0;
}

void Avg3DPlugin::handshake()
{
	HostBuffer<char> data;
	SimpleSerializer::serialize(data, resolution, h, needDensity, needMomentum, needForce);

	MPI_Check( MPI_Send(data.hostPtr(), data.size(), MPI_BYTE, rank, id, interComm) );

	debug2("Plugin %s was set up to sample%s%s%s for the following PVs: %s. Resolution %dx%dx%d", name.c_str(),
			needDensity ? " density" : "", needMomentum ? " momentum" : "", needForce ? " force" : "", pvNames.c_str(),
			resolution.x, resolution.y, resolution.z);
}





Avg3DDumper::Avg3DDumper(std::string name, std::string path, int3 nranks3D) :
		PostprocessPlugin(name), path(path), nranks3D(nranks3D) { }

void Avg3DDumper::handshake()
{
	HostBuffer<char> buf(1000);
	MPI_Check( MPI_Recv(buf.hostPtr(), buf.size(), MPI_BYTE, rank, id, interComm, MPI_STATUS_IGNORE) );
	SimpleSerializer::deserialize(buf, resolution, h, needDensity, needMomentum, needForce);
	int totalPoints = resolution.x * resolution.y * resolution.z;

	std::vector<std::string> channelNames;
	std::vector<XDMFDumper::ChannelType> channelTypes;

	// For current time
	data.resize(sizeof(float));
	if (needDensity)
	{
		channelNames.push_back("density");
		channelTypes.push_back(XDMFDumper::ChannelType::Scalar);
		density.resize(totalPoints);
	}
	if (needMomentum)
	{
		channelNames.push_back("momentum");
		channelTypes.push_back(XDMFDumper::ChannelType::Vector);
		momentum.resize(totalPoints);
	}
	if (needForce)
	{
		channelNames.push_back("force");
		channelTypes.push_back(XDMFDumper::ChannelType::Vector);
		force.resize(totalPoints);
	}

	float t;
	data.resize(SimpleSerializer::totSize(t, density, momentum, force));

	debug2("Plugin %s was set up to dump%s%s%s. Resolution %dx%dx%d. Path %s", name.c_str(),
			needDensity ? " density" : "", needMomentum ? " momentum" : "", needForce ? " force" : "",
			resolution.x, resolution.y, resolution.z, path.c_str());

	dumper = new XDMFDumper(comm, nranks3D, path, resolution, h, channelNames, channelTypes);

	size = data.size();
}

void Avg3DDumper::deserialize(MPI_Status& stat)
{
	float t;
	SimpleSerializer::deserialize(data, t, density, momentum, force);

	std::vector<const float*> channels;
	if (needDensity)  channels.push_back(density.hostPtr());
	if (needMomentum) channels.push_back((const float*)momentum.hostPtr());
	if (needForce)    channels.push_back((const float*)force.hostPtr());

	debug2("Plugin %s will dump right now", name.c_str());
	dumper->dump(channels, t);
}

