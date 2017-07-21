#include <core/celllist.h>
#include <core/particle_vector.h>
#include <core/helper_math.h>

#include <core/mpi/particle_redistributor.h>
#include <core/mpi/valid_cell.h>

#include <vector>
#include <thread>
#include <algorithm>

__global__ void getExitingParticles(float4* xyzouvwo,
		CellListInfo cinfo, const uint* __restrict__ cellsStartSize,
		const int64_t dests[27], int counts[27])
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	int cid;
	int cx, cy, cz;
	const int3 ncells = cinfo.ncells;

	bool valid = isValidCell(cid, cx, cy, cz, gid, blockIdx.y, cinfo);

	if (!valid) return;

	// The following is called for every outer cell and exactly once for each
	//
	// Now for each cell we check its every particle if it needs to move

	int2 start_size = cinfo.decodeStartSize(cellsStartSize[cid]);

#pragma unroll 2
	for (int i = 0; i < start_size.y; i++)
	{
		const int srcId = start_size.x + i;
		const float4 coo = xyzouvwo[2*srcId];
		const float4 vel = xyzouvwo[2*srcId+1];

		int3 code = cinfo.getCellIdAlongAxis<false>(make_float3(coo));

		if (code.x < 0) code.x = 0;
		else if (code.x >= ncells.x) code.x = 2;
		else code.x = 1;

		if (code.y < 0) code.y = 0;
		else if (code.y >= ncells.y) code.y = 2;
		else code.y = 1;

		if (code.z < 0) code.z = 0;
		else if (code.z >= ncells.z) code.z = 2;
		else code.z = 1;

		if (code.x*code.y*code.z != 1) // this means that the particle has to leave
		{
			const int bufId = (code.z*3 + code.y)*3 + code.x;
			const float4 shift{ cinfo.domainSize.x*(code.x-1),
								cinfo.domainSize.y*(code.y-1),
								cinfo.domainSize.z*(code.z-1), 0 };

			int myid = atomicAdd(counts + bufId, 1);

			const int dstInd = 2*myid;

			float4* addr = (float4*)dests[bufId];
			float4 newcoo = coo - shift;
			newcoo.w = coo.w;
			addr[dstInd + 0] = newcoo;
			addr[dstInd + 1] = vel;

			// mark the particle as exited to assist cell-list building
			xyzouvwo[2*srcId] = make_float4(-1000.0f, -1000.0f, -1000.0f, coo.w);
		}
	}
}

void ParticleRedistributor::attach(ParticleVector* pv, CellList* cl)
{
	particles.push_back(pv);
	cellLists.push_back(cl);

	const double ndens = (double)pv->local()->size() / (cl->ncells.x * cl->ncells.y * cl->ncells.z * cl->rc*cl->rc*cl->rc);
	const int maxdim = std::max({cl->domainSize.x, cl->domainSize.y, cl->domainSize.z});

	// Sizes of buffers. 0 is side, 1 is edge, 2 is corner
	const int sizes[3] = { (int)(ndens * maxdim*maxdim + 128), (int)(ndens * maxdim + 128), (int)(ndens + 128) };

	auto helper = new ExchangeHelper(pv->name, sizeof(Particle), sizes);
	helpers.push_back(helper);
}

void ParticleRedistributor::redistribute()
{
	init();
	finalize();
}

void ParticleRedistributor::prepareData(int id)
{
	auto pv = particles[id];
	auto cl = cellLists[id];
	auto helper = helpers[id];

	debug2("Preparing %s leaving particles on the device", pv->name.c_str());

	helper->bufSizes.pushStream(defStream);
	helper->bufSizes.clearDevice();
	helper->bufSizes.popStream();

	const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
	const int nthreads = 32;
	if (pv->local()->size() > 0)
		getExitingParticles<<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 6, 1),  dim3(nthreads, 1, 1), 0, helper->stream >>>
					( (float4*)pv->local()->coosvels.devPtr(), cl->cellInfo(), cl->cellsStartSize.devPtr(), (int64_t*)helper->sendAddrs.devPtr(), helper->bufSizes.devPtr() );
}

void ParticleRedistributor::combineAndUploadData(int id)
{
	auto pv = particles[id];
	auto helper = helpers[id];

	int oldsize = pv->local()->size();
	pv->local()->resize(oldsize + helper->recvOffsets[27], resizePreserve);

	auto ptr = pv->local()->coosvels.devPtr() + oldsize;

	for (int i=0; i < 27; i++)
	{
		const int msize = helper->recvOffsets[i+1] - helper->recvOffsets[i];
		if (msize > 0)
			CUDA_Check( cudaMemcpyAsync(ptr + helper->recvOffsets[i], helper->recvBufs[i].hostPtr(),
					msize*sizeof(Particle), cudaMemcpyHostToDevice, helper->stream) );
	}
}
