#include "datatypes.h"
#include "containers.h"
#include "redistributor.h"
#include "celllist.h"
#include "logger.h"

#include <vector>
#include <thread>
#include <algorithm>
#include <unistd.h>

template<typename Transform>
__global__ void getExitingParticles(const int* __restrict__ cellsStart, const float4* __restrict__ xyzouvwo, const float4* __restrict__ accs,
		const int3 ncells, const int totcells, const float3 length, const float3 domainStart,
		const int64_t __restrict__ dests[27], int counts[27], Transform transform)
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int variant = blockIdx.y;
	int cid;
	int cx, cy, cz;

	// Select all the boundary cells WITOUT duplicates
	bool valid = true;

	if (variant <= 1)  // x
	{
		if (gid >= ncells.y*ncells.z) valid = false;
		cx = variant * (ncells.x - 1);
		cy = gid % ncells.y;
		cz = gid / ncells.y;
		cid = encode(cx, cy, cz, ncells);
	}
	else if (variant <= 3)  // y
	{
		if (gid >= ncells.x*ncells.z) valid = false;
		cx = gid % ncells.x;
		cy = (variant - 2) * (ncells.y - 1);
		cz = gid / ncells.x;
		cid = encode(cx, cy, cz, ncells);
	}
	else   // z
	{
		if (gid >= ncells.x*ncells.y) valid = false;
		cx = gid % ncells.x;
		cy = gid / ncells.x;
		cz = (variant - 4) * (ncells.z - 1);
		cid = encode(cx, cy, cz, ncells);
	}

	valid &= cid < totcells;

	// Find side codes
	if (cx == 0) cx = 0;
	else if (cx == ncells.x-1) cx = 2;
	else cx = 1;

	if (cy == 0) cy = 0;
	else if (cy == ncells.y-1) cy = 2;
	else cy = 1;

	if (cz == 0) cz = 0;
	else if (cz == ncells.z-1) cz = 2;
	else cz = 1;

	// Exclude cells already covered by other variants
	if ( (variant == 0 || variant == 1) && (cz != 1 || cy != 1) ) valid = false;
	if ( (variant == 2 || variant == 3) && (cz != 1) ) valid = false;

	if (!valid) return;

	// The following is called for every outer cell and exactly once for each
	//
	// Now for each cell we check every particle if it needs to move

	int2 start_size = valid ? decodeStartSize(cellsStart[cid]) : make_int2(0, 0);

	for (int i = 0; i < start_size.y; i++)
	{
		const int srcId = start_size.x + i;
		const float4 coo = xyzouvwo[2*srcId];
		const float4 vel = xyzouvwo[2*srcId+1];
		const float4 acc = accs[srcId];

		float4 newcoo = coo, newvel = vel;
		transform(newcoo, newvel, acc);

		int px = getCellIdAlongAxis<false>(newcoo.x, domainStart.x, ncells.x, 1.0f);
		int py = getCellIdAlongAxis<false>(newcoo.y, domainStart.y, ncells.y, 1.0f);
		int pz = getCellIdAlongAxis<false>(newcoo.z, domainStart.z, ncells.z, 1.0f);

		if (px <= -1) px = 0;
		else if (px >= ncells.x) px = 2;
		else px = 1;

		if (py <= -1) py = 0;
		else if (py >= ncells.y) py = 2;
		else py = 1;

		if (pz <= -1) pz = 0;
		else if (pz >= ncells.z) pz = 2;
		else pz = 1;

		for (int ix = min(px, 1); ix <= max(px, 1); ix++)
			for (int iy = min(py, 1); iy <= max(py, 1); iy++)
				for (int iz = min(pz, 1); iz <= max(pz, 1); iz++)
				{
					if (ix == 1 && iy == 1 && iz == 1) continue;

					const int bufId = (iz*3 + iy)*3 + ix;
					const float4 shift{ length.x*(ix-1), length.y*(iy-1), length.z*(iz-1), 0 };

					int myid = atomicAdd(counts + bufId, 1);

					const int dstInd = 3*myid;

					float4* addr = (float4*)dests[bufId];
					addr[dstInd + 0] = coo - shift;
					addr[dstInd + 1] = vel;
					addr[dstInd + 2] = acc;
				}

	}
}

Redistributor::Redistributor(MPI_Comm& comm) : nActiveNeighbours(26)
{
	MPI_Check( MPI_Comm_dup(comm, &redComm));

	int dims[3], periods[3], coords[3];
	MPI_Check( MPI_Cart_get (redComm, 3, dims, periods, coords) );
	MPI_Check( MPI_Comm_rank(redComm, &myrank));

	MPI_Check( MPI_Type_contiguous(sizeof(PandA), MPI_BYTE, &mpiPandAType) );
	MPI_Check( MPI_Type_commit(&mpiPandAType) );

	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3, (i/3) % 3, i/9 };

		int coordsNeigh[3];
		for(int c = 0; c < 3; ++c)
			coordsNeigh[c] = coords[c] + d[c];

		MPI_Check( MPI_Cart_rank(redComm, coordsNeigh, dir2rank + i) );
	}
}

void Redistributor::attach(ParticleVector* pv, int ndens)
{
	particleVectors.push_back(pv);

	helpers.resize(helpers.size() + 1);
	auto& helper = helpers[helpers.size() - 1];

	helper.sendAddrs.resize(27);
	helper.counts.resize(27);

	const int maxdim = std::max({pv->ncells.x, pv->ncells.y, pv->ncells.z});

	CUDA_Check( cudaStreamCreateWithPriority(&helper.stream, cudaStreamNonBlocking, -10) );

	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

		int c = std::abs(d[0]) + std::abs(d[1]) + std::abs(d[2]);
		if (c > 0)
		{
			helper.sendBufs[i].resize( 3 * ndens * pow(maxdim, 3 - c) );
			helper.sendAddrs[i] = (float4*)helper.sendBufs[i].devdata;
		}
	}
	helper.sendAddrs.synchronize(synchronizeDevice);
}

void Redistributor::redistribute(float dt)
{
	for (int i=0; i<particleVectors.size(); i++)
		__identify(i, dt);

	for (int i=0; i<particleVectors.size(); i++)
	{
		CUDA_Check( cudaStreamSynchronize(helpers[i].stream) );
		send(i);
	}

	for (int i=0; i<particleVectors.size(); i++)
		receive(i);
}

void Redistributor::__identify(int n, float dt)
{
	auto& pv = particleVectors[n];
	auto& helper = helpers[n];

	const int maxdim = std::max({pv->ncells.x, pv->ncells.y, pv->ncells.z});
	const int nthreads = 32;
	helper.counts.clear(helper.stream);

	auto integrate = [dt] __device__ (float4& x, float4& v, float4 a) {
		v.x += a.x*dt;
		v.y += a.y*dt;
		v.z += a.z*dt;

		x.x += v.x*dt;
		x.y += v.y*dt;
		x.z += v.z*dt;
	};

	debug("Preparing %d-th leaving particles on the device", n);
	getExitingParticles<<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 6, 1),  dim3(nthreads, 1, 1), 0, helper.stream >>>
			(pv->cellsStart.devdata, (float4*)pv->coosvels.devdata, (float4*)pv->accs.devdata,
			 pv->ncells, pv->totcells, pv->length, pv->domainStart,
			 (int64_t*)helper.sendAddrs.devdata, helper.counts.devdata, integrate);
}

void Redistributor::send(int n)
{
	auto& pv = particleVectors[n];
	auto& helper = helpers[n];

	helper.counts.synchronize(synchronizeHost, helper.stream);
	debug("Downloading %d-th leaving particles", n);
	for (int i=0; i<27; i++)
		if (i != 13)
			CUDA_Check( cudaMemcpyAsync(helper.sendBufs[i].hostdata, helper.sendBufs[i].devdata,
					helper.counts[i]*sizeof(PandA), cudaMemcpyDeviceToHost, helper.stream) );
	CUDA_Check( cudaStreamSynchronize(helper.stream) );

	MPI_Request req;
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			debug("Sending %d-th redistribution to rank %d in dircode %d, size %d", n, dir2rank[i], i, helper.counts[i]);
			MPI_Check( MPI_Isend(helper.sendBufs[i].hostdata, helper.counts[i], mpiPandAType, dir2rank[i], n, redComm, &req) );
		}
}

void Redistributor::receive(int n)
{
	auto& helper = helpers[n];
	auto& pv = particleVectors[n];

	int cur = 0;
	helper.recvBuf.resize(0);

	for (int i=0; i<nActiveNeighbours; i++)
	{
		MPI_Status stat;
		int recvd = 0;
		while (recvd == 0)
		{
			MPI_Check( MPI_Iprobe(MPI_ANY_SOURCE, n, redComm, &recvd, &stat) );
			//if (recvd == 0) usleep(10);
		}

		int msize;
	    MPI_Check( MPI_Get_count(&stat, mpiPandAType, &msize) );
	    helper.recvBuf.resize(helper.recvBuf.size + msize, resizePreserve);

		debug("Receiving %d-th redistribution from rank %d, size %d", n, stat.MPI_SOURCE, msize);
		MPI_Check( MPI_Recv(helper.recvBuf.hostdata+cur, msize, mpiPandAType, stat.MPI_SOURCE, n, redComm, &stat) );

		cur += msize;
	}

	int total = cur;
	helper.recvPartBuf.resize(total);
	helper.recvAccBuf. resize(total);

	for (int i=0; i<total; i++)
	{
		helper.recvPartBuf[i] = helper.recvBuf[i].p;
		helper.recvAccBuf [i] = helper.recvBuf[i].a;
	}

	int oldsize = pv->np;
	pv->resize(oldsize + total, resizePreserve, helper.stream);

	CUDA_Check( cudaMemcpyAsync(pv->coosvels.devdata + oldsize, helper.recvPartBuf.hostdata, total*sizeof(Particle),     cudaMemcpyHostToDevice, helper.stream) );
	CUDA_Check( cudaMemcpyAsync(pv->accs.    devdata + oldsize, helper.recvAccBuf. hostdata, total*sizeof(Acceleration), cudaMemcpyHostToDevice, helper.stream) );

	CUDA_Check( cudaStreamSynchronize(helper.stream) );
}


