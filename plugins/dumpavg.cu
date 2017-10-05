#include "dumpavg.h"
#include "simple_serializer.h"
#include "string2vector.h"

#include <core/simulation.h>
#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/cuda_common.h>

__global__ void sample(PVview pvView, CellListInfo cinfo, float* avgDensity, float3* avgMomentum, float3* avgForce)
{
	const int pid = threadIdx.x + blockIdx.x*blockDim.x;
	if (pid >= pvView.size) return;

	const float4 coo = pvView.particles[2*pid];
	const int cid = cinfo.getCellId(coo);

	if (avgDensity != nullptr)
		atomicAdd(avgDensity+cid, pvView.mass);

	if (avgMomentum != nullptr)
	{
		const float3 momentum = make_float3(pvView.particles[2*pid+1] * pvView.mass);
		atomicAdd( (float*)(avgMomentum + cid)  , momentum.x);
		atomicAdd( (float*)(avgMomentum + cid)+1, momentum.y);
		atomicAdd( (float*)(avgMomentum + cid)+2, momentum.z);
	}

	if (avgForce != nullptr)
	{
		const float3 frc = make_float3(pvView.forces[pid]);
		atomicAdd( (float*)(avgForce + cid)  , frc.x);
		atomicAdd( (float*)(avgForce + cid)+1, frc.y);
		atomicAdd( (float*)(avgForce + cid)+2, frc.z);
	}
}

__global__ void scaleVec(int n, float3* vectorField, const float* density)
{
	const int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < n)
		vectorField[id] /= (density[id] + 1e-8f);
}

__global__ void scaleDensity(int n, float* density, const float factor)
{
	const int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < n)
		density[id] *= factor;
}

Avg3DPlugin::Avg3DPlugin(std::string name, std::string pvNames, int sampleEvery, int dumpEvery, float3 binSize,
			bool needMomentum, bool needForce) :
	SimulationPlugin(name, true), pvNames(pvNames),
	sampleEvery(sampleEvery), dumpEvery(dumpEvery), binSize(binSize),
	needDensity(true), needMomentum(needMomentum), needForce(needForce),
	nSamples(0)
{ }

void Avg3DPlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
	SimulationPlugin::setup(sim, comm, interComm);

	// TODO: this should be reworked if the domains are allowed to have different size
	resolution = make_int3( floorf(sim->localDomainSize / binSize) );
	binSize = sim->localDomainSize / make_float3(resolution);

	const int total = resolution.x * resolution.y * resolution.z;
	if (needDensity)  density .resize(total, 0);
	if (needMomentum) momentum.resize(total, 0);
	if (needForce)    force   .resize(total, 0);

	auto splitPvNames = splitByDelim(pvNames);

	density.clear(0);
	momentum.clear(0);
	force.clear(0);

	for (auto& nm : splitPvNames)
	{
		auto pv = sim->getPVbyName(nm);
		if (pv == nullptr)
			die("No such particle vector registered: %s", nm.c_str());

		particleVectors.push_back(pv);
	}

	info("Plugin %s initialized for the following particle vectors: %s", name.c_str(), pvNames.c_str());
}



void Avg3DPlugin::afterIntegration(cudaStream_t stream)
{
	if (currentTimeStep % sampleEvery != 0 || currentTimeStep == 0) return;

	debug2("Plugin %s is sampling now", name.c_str());

	for (auto pv : particleVectors)
	{
		CellListInfo cinfo(binSize, pv->localDomainSize);
		auto pvView = create_PVview(pv, pv->local());

		sample<<< (pvView.size+127) / 128, 128, 0, stream >>> (
				pvView, cinfo,
				needDensity  ? density .devPtr() : nullptr,
				needMomentum ? momentum.devPtr() : nullptr,
				needForce    ? force   .devPtr() : nullptr );
	}

	nSamples++;
}

void Avg3DPlugin::serializeAndSend(cudaStream_t stream)
{
	if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

	// Order is important here! First mom and frc, only then dens
	if (needMomentum)
	{
		int sz = momentum.size();
		scaleVec<<< (sz+127)/128, 128, 0, stream >>> (sz, momentum.devPtr(), density.devPtr());
		momentum.downloadFromDevice(stream);
		momentum.clearDevice(stream);
	}

	if (needForce)
	{
		int sz = force.size();
		scaleVec<<< (sz+127)/128, 128, 0, stream >>> (sz, force.devPtr(), density.devPtr());
		force.downloadFromDevice(stream);
		force.clearDevice(stream);
	}

	if (needDensity)
	{
		int sz = density.size();
		scaleDensity<<< (sz+127)/128, 128, 0, stream >>> ( sz, density.devPtr(), 1.0 / (nSamples * binSize.x*binSize.y*binSize.z) );
		density.downloadFromDevice(stream);
		density.clearDevice(stream);
	}

	debug2("Plugin %s is sending now data", name.c_str());
	SimpleSerializer::serialize(sendBuffer, currentTime, density, momentum, force);
	send(sendBuffer.data(), sendBuffer.size());

	nSamples = 0;
}

void Avg3DPlugin::handshake()
{
	std::vector<char> data;
	SimpleSerializer::serialize(data, sim->nranks3D, resolution, binSize, needDensity, needMomentum, needForce);
	send(data.data(), data.size());

	debug2("Plugin %s was set up to sample%s%s%s for the following PVs: %s. Local resolution %dx%dx%d", name.c_str(),
			needDensity ? " density" : "", needMomentum ? " momentum" : "", needForce ? " force" : "", pvNames.c_str(),
			resolution.x, resolution.y, resolution.z);
}




Avg3DDumper::Avg3DDumper(std::string name, std::string path) :
		PostprocessPlugin(name), path(path) { }

void Avg3DDumper::handshake()
{
	auto req = waitData();
	MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
	recv();

	SimpleSerializer::deserialize(data, nranks3D, resolution, h, needDensity, needMomentum, needForce);
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

	debug2("Plugin %s was set up to dump%s%s%s. Resolution %dx%dx%d. Path %s", name.c_str(),
			needDensity ? " density" : "", needMomentum ? " momentum" : "", needForce ? " force" : "",
			resolution.x, resolution.y, resolution.z, path.c_str());

	dumper = new XDMFDumper(comm, nranks3D, path, resolution, h, channelNames, channelTypes);
}

void Avg3DDumper::deserialize(MPI_Status& stat)
{
	float t;
	SimpleSerializer::deserialize(data, t, density, momentum, force);

	std::vector<const float*> channels;
	if (needDensity)  channels.push_back(density.data());
	if (needMomentum) channels.push_back((const float*)momentum.data());
	if (needForce)    channels.push_back((const float*)force.data());

	debug2("Plugin %s will dump right now", name.c_str());
	dumper->dump(channels, t);
}

