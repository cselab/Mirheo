#include "average_relative_flow.h"

#include <core/utils/kernel_launch.h>
#include <core/simulation.h>
#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>

#include "simple_serializer.h"
#include "sampling_helpers.h"


__global__ void sampleRelative(
		PVview pvView, CellListInfo cinfo,
		float* avgDensity,
		ChannelsInfo channelsInfo,
		float3 relativePoint)
{
	const int pid = threadIdx.x + blockIdx.x*blockDim.x;
	if (pid >= pvView.size) return;

	Particle p(pvView.particles, pid);
	p.r -= relativePoint;

	int3 cid3 = cinfo.getCellIdAlongAxes<false>(p.r);
	cid3 = (cid3 + cinfo.ncells) % cinfo.ncells;
	const int cid = cinfo.encode(cid3);

	atomicAdd(avgDensity + cid, 1);

	sampleChannels(pid, cid, channelsInfo);
}


AverageRelative3D::AverageRelative3D(
		std::string name,
		std::string pvName,
		std::vector<std::string> channelNames, std::vector<Average3D::ChannelType> channelTypes,
		int sampleEvery, int dumpEvery, float3 binSize,
		std::string relativeOVname, int relativeID) :

		Average3D(name, pvName, channelNames, channelTypes, sampleEvery, dumpEvery, binSize),
		relativeOVname(relativeOVname), relativeID(relativeID)

{	}

void AverageRelative3D::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
	Average3D::setup(sim, comm, interComm);

	localChannels.resize(channelsInfo.n + 1);
	localChannels[0].resize(density.size());
	density.resize_anew(density.size() * nranks);
	density.clear(0);

	for (int i=0; i<channelsInfo.n; i++)
	{
		localChannels[i + 1].resize(channelsInfo.average[i].size());
		channelsInfo.average[i].resize_anew(channelsInfo.average[i].size() * nranks);
		channelsInfo.average[i].clear(0);
		channelsInfo.averagePtrs[i] = channelsInfo.average[i].devPtr();
	}

	channelsInfo.averagePtrs.uploadToDevice(0);

	// Relative stuff
	relativeOV = sim->getOVbyNameOrDie(relativeOVname);

	if ( !relativeOV->local()->extraPerObject.checkChannelExists("motions") )
		die("Only rigid objects are support for relative flow, but got OV '%s'", relativeOV->name.c_str());

	int locsize = relativeOV->local()->nObjects;
	int totsize;

	MPI_Check( MPI_Reduce(&locsize, &totsize, 1, MPI_INT, MPI_SUM, 0, comm) );

	if (rank == 0 && relativeID >= totsize)
		die("To few objects in OV '%s' (only %d); but requested id %d",
				relativeOV->name.c_str(), totsize, relativeID);
}


void AverageRelative3D::afterIntegration(cudaStream_t stream)
{
	if (currentTimeStep % sampleEvery != 0 || currentTimeStep == 0) return;

	debug2("Plugin %s is sampling now", name.c_str());

	float3 relativeParams[2] = {make_float3(0.0f), make_float3(0.0f)};

	// Find and broadcast the position and velocity of the relative object
	MPI_Request req;
	MPI_Check( MPI_Irecv(relativeParams, 6, MPI_FLOAT, MPI_ANY_SOURCE, 22, comm, &req) );

	auto ids     = relativeOV->local()->extraPerObject.getData<int>("ids");
	auto motions = relativeOV->local()->extraPerObject.getData<RigidMotion>("motions");

	ids    ->downloadFromDevice(stream, false);
	motions->downloadFromDevice(stream, true);

	for (int i=0; i < ids->size(); i++)
	{
		if ((*ids)[i] == relativeID)
		{
			float3 params[2] = { make_float3( (*motions)[i].r ),
					make_float3( (*motions)[i].vel ) };

			params[0] = sim->domain.local2global(params[0]);

			for (int r = 0; r < nranks; r++)
				MPI_Send(&params, 6, MPI_FLOAT, r, 22, comm);

			break;
		}
	}

	MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );

	relativeParams[0] = sim->domain.global2local(relativeParams[0]);

	CellListInfo cinfo(binSize, pv->domain.globalSize);
	PVview pvView(pv, pv->local());
	ChannelsInfo gpuInfo(channelsInfo, pv, stream);

	const int nthreads = 128;
	SAFE_KERNEL_LAUNCH(
			sampleRelative,
			getNblocks(pvView.size, nthreads), nthreads, 0, stream,
			pvView, cinfo, density.devPtr(), gpuInfo, relativeParams[0]);

	averageRelativeVelocity += relativeParams[1];

	nSamples++;
}


void AverageRelative3D::extractLocalBlock()
{
	int locId = 0;

	auto oneChannel = [&locId, this] (PinnedBuffer<float>& channel, Average3D::ChannelType type, float scale) {

		MPI_Check( MPI_Allreduce(MPI_IN_PLACE, channel.hostPtr(), channel.size(), MPI_FLOAT, MPI_SUM, comm) );
		//MPI_Check( MPI_Bcast(channel.hostPtr(), channel.size(), MPI_FLOAT, 0, comm) );

		int ncomponents = 3;
		if (type == Average3D::ChannelType::Scalar)  ncomponents = 1;
		if (type == Average3D::ChannelType::Tensor6) ncomponents = 6;

		int3 globalResolution = resolution * sim->nranks3D;
		int3 rank3D = sim->rank3D;

		float factor;
		int dstId=0;
		for (int k = rank3D.z*resolution.z; k < (rank3D.z+1)*resolution.z; k++)
			for (int j = rank3D.y*resolution.y; j < (rank3D.y+1)*resolution.y; j++)
				for (int i = rank3D.x*resolution.x; i < (rank3D.x+1)*resolution.x; i++)
				{
					int scalId = (k*globalResolution.y*globalResolution.x + j*globalResolution.x + i);
					int srcId = ncomponents * scalId;
					for (int c=0; c<ncomponents; c++)
					{
						if (scale < 0.0f) factor = 1.0f / density[scalId];
						else factor = scale;

						localChannels[locId][dstId++] = channel[srcId] * factor;
						srcId++;
					}
				}

		locId++;
	};

	// Order is important! Density comes first
	oneChannel(density, Average3D::ChannelType::Scalar, pv->mass / (nSamples * binSize.x*binSize.y*binSize.z));

	for (int i=0; i<channelsInfo.n; i++)
		oneChannel(channelsInfo.average[i], channelsInfo.types[i], -1);
}

void AverageRelative3D::serializeAndSend(cudaStream_t stream)
{
	if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

	for (int i=0; i<channelsInfo.n; i++)
	{
		auto& data = channelsInfo.average[i];

		if (channelsInfo.names[i] == "velocity")
		{
			const int nthreads = 128;

			SAFE_KERNEL_LAUNCH(
					correctVelocity,
					getNblocks(data.size() / 3, nthreads), nthreads, 0, stream,
					data.size() / 3, (float3*)data.devPtr(), density.devPtr(), averageRelativeVelocity / (float) nSamples);

			averageRelativeVelocity = make_float3(0);
		}
	}

	density.downloadFromDevice(stream, true);
	density.clearDevice(stream);

	for (auto& data : channelsInfo.average)
	{
		data.downloadFromDevice(stream, false);
		data.clearDevice(stream);
	}


	extractLocalBlock();
	nSamples = 0;


	// Calculate total size for sending
	int totalSize = SimpleSerializer::totSize(currentTime);
	for (auto& ch : localChannels)
		totalSize += SimpleSerializer::totSize(ch);

	// Now allocate the sending buffer and pack everything into it
	debug2("Plugin %s is packing now data", name.c_str());
	sendBuffer.resize(totalSize);

	SimpleSerializer::serialize(sendBuffer.data(), currentTime);
	int currentSize = SimpleSerializer::totSize(currentTime);

	for (auto& ch : localChannels)
	{
		SimpleSerializer::serialize(sendBuffer.data() + currentSize, ch);
		currentSize += SimpleSerializer::totSize(ch);
	}

	send(sendBuffer);
}

