#include "particle_redistributor.h"

#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/cuda_common.h>

#include <core/mpi/valid_cell.h>

template<bool QUERY=false>
__global__ void getExitingParticles(float4* coosvels,
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
		Particle p(coosvels, srcId);

		int3 code = cinfo.getCellIdAlongAxis<false>(make_float3(p.r));

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
			const float3 shift{ cinfo.localDomainSize.x*(code.x-1),
								cinfo.localDomainSize.y*(code.y-1),
								cinfo.localDomainSize.z*(code.z-1) };
			p.r -= shift;

			int myid = atomicAdd(counts + bufId, 1);

			if (QUERY) continue;

			const int dstInd = myid;
			float4* addr = (float4*)dests[bufId];
			addr[2*dstInd + 0] = p.r2Float4();
			addr[2*dstInd + 1] = p.u2Float4();

			// mark the particle as exited to assist cell-list building
			coosvels[2*srcId] = Float3_int(make_float3(-1e5), p.i1).toFloat4();
		}
	}
}

void ParticleRedistributor::attach(ParticleVector* pv, CellList* cl)
{
	particles.push_back(pv);
	cellLists.push_back(cl);

	if (dynamic_cast<PrimaryCellList*>(cl) == nullptr)
		die("Redistributor (for %s) should be used with the primary cell-lists only!", pv->name.c_str());

	auto helper = new ExchangeHelper(pv->name, sizeof(Particle));
	helpers.push_back(helper);

	info("Particle redistributor takes pv %s, base tag %d", pv->name.c_str(), tagByName(pv->name));
}

void ParticleRedistributor::prepareData(int id, cudaStream_t stream)
{
	auto pv = particles[id];
	auto cl = cellLists[id];
	auto helper = helpers[id];

	debug2("Preparing %s leaving particles on the device", pv->name.c_str());

	const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
	const int nthreads = 64;
	if (pv->local()->size() > 0)
	{
		helper->sendBufSizes.clear(stream);
		getExitingParticles<true>  <<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 6, 1),  dim3(nthreads, 1, 1), 0, stream>>> (
				(float4*)pv->local()->coosvels.devPtr(), cl->cellInfo(), cl->cellsStartSize.devPtr(),
				(int64_t*)helper->sendAddrs.devPtr(), helper->sendBufSizes.devPtr() );

		helper->sendBufSizes.downloadFromDevice(stream);
		helper->resizeSendBufs(stream);

		helper->sendBufSizes.clearDevice(stream);
		getExitingParticles<false> <<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 6, 1),  dim3(nthreads, 1, 1), 0, stream>>> (
				(float4*)pv->local()->coosvels.devPtr(), cl->cellInfo(), cl->cellsStartSize.devPtr(),
				(int64_t*)helper->sendAddrs.devPtr(), helper->sendBufSizes.devPtr() );
	}
}

void ParticleRedistributor::combineAndUploadData(int id, cudaStream_t stream)
{
	auto pv = particles[id];
	auto helper = helpers[id];

	int oldsize = pv->local()->size();
	pv->local()->resize(oldsize + helper->recvOffsets[27], stream, ResizeKind::resizePreserve);

	auto hptr = pv->local()->coosvels.hostPtr() + oldsize;
	auto dptr = pv->local()->coosvels.devPtr() + oldsize;

	for (int i=0; i < 27; i++)
	{
		const int msize = helper->recvOffsets[i+1] - helper->recvOffsets[i];
		if (msize > 0)
			memcpy(hptr + helper->recvOffsets[i], helper->recvBufs[i].hostPtr(), msize*sizeof(Particle));
	}

	CUDA_Check( cudaMemcpyAsync(dptr, hptr, helper->recvOffsets[27]*sizeof(Particle), cudaMemcpyHostToDevice, stream) );

	// The PV has changed significantly, need to update che cell-lists now
	pv->local()->changedStamp++;
}
