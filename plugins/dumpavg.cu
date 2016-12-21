#include "dumpavg.h"

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

template<typename ValType, typename SampleKernel>
__global__ void sample(const int * const __restrict__ cellsStart, ValType* res, SampleKernel kernel)
{
	const int3 ccoos = {
			threadIdx.x + blockIdx.x*blockDim.x,
			threadIdx.y + blockIdx.y*blockDim.y,
			threadIdx.z + blockIdx.z*blockDim.z};

	if (ccoos.x < ncells.x && ccoos.y < ncells.y && ccoos.z < ncells.z)
	{
		const int cid = CellList::encode(ccoos.x, ccoos.y, ccoos.z, ncells);
		const int2 start_size = CellList::decodeStartSize(cellsStart[cid]);

		kernel(res[cid], start_size.x, start_size.y);
	}
}

__global__ void scale(int n, float a, float* res)
{
	const uint id = threadIdx.x + blockIdx.x*blockDim.x;

	if (id < n)
	{
		res[id] *= a;
	}
}

DumpAvg3D::DumpAvg3D(Simulation* sim, std::string pvNames, int sampleEvery, int3 resolution, bool needDensity, bool needVelocity, bool needForce, std::string namePrefix) :
	Plugin(sim), sampleEvery(sampleEvery), resolution(resolution), density(needDensity), velocity(needVelocity), force(needForce), namePrefix(namePrefix)
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
			die("No such particles vector registered: %s", nm.c_str());

		particleVectors.push_back(sim->particleVectors[pvIter->second]);
	}

	cellList.resize(total+1);

#ifdef __MORTON__
	die("DumpAvg3d doesn't support morton indexing (yet)");
#endif
}


void DumpAvg3D::afterIntegration(cudaStream_t stream)
{
	if (needDensity)  density .clear();
	if (needVelocity) velocity.clear();
	if (needForce)    force   .clear();

	for (auto pv : particleVectors)
	{

		buildCellList(pv, cellList.devdata, stream);
		if (needDensity)
		{
			float mass = pv->mass;

			auto densSampler = [=] __device__ (float* dens, int start, int count) {

			};

	}
}
