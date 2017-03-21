#include <core/particle_vector.h>
#include <core/object_vector.h>
#include <core/halo_exchanger.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/non_cached_rw.h>

#include <vector>
#include <algorithm>
#include <limits>

__device__ inline bool isValidCell(int& cid, int& cx, int& cy, int& cz, int gid, int variant, CellListInfo cinfo)
{
	const int3 ncells = cinfo.ncells;

	bool valid = true;

	if (variant <= 1)  // x
	{
		if (gid >= ncells.y * ncells.z) valid = false;
		cx = variant * (ncells.x - 1);
		cy = gid % ncells.y;
		cz = gid / ncells.y;
	}
	else if (variant <= 3)  // y
	{
		if (gid >= ncells.x * ncells.z) valid = false;
		cx = gid % ncells.x;
		cy = (variant - 2) * (ncells.y - 1);
		cz = gid / ncells.x;
	}
	else   // z
	{
		if (gid >= ncells.x * ncells.y) valid = false;
		cx = gid % ncells.x;
		cy = gid / ncells.x;
		cz = (variant - 4) * (ncells.z - 1);
	}

	cid = cinfo.encode(cx, cy, cz);

	valid &= cid < cinfo.totcells;

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

	return valid;
}

__global__ void getHalos(const float4* __restrict__ coosvels, const CellListInfo cinfo, const uint* __restrict__ cellsStartSize,
		const int64_t dests[27], int counts[27])
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	int cid;
	int cx, cy, cz;

	bool valid = isValidCell(cid, cx, cy, cz, gid, blockIdx.y, cinfo);

	if (__all(!valid) && tid > 27) return;

	int2 start_size = valid ? cinfo.decodeStartSize(cellsStartSize[cid]) : make_int2(0, 0);

	// Use shared memory to decrease number of global atomics
	// We're sending to max 7 halos (corner)
	short validHalos[7];
	int haloOffset[7] = {};

	int current = 0;

	// Total number of elements written to halos by this block
	__shared__ int blockSum[27];
	if (tid < 27) blockSum[tid] = 0;

	__syncthreads();

	for (int ix = min(cx, 1); ix <= max(cx, 1); ix++)
		for (int iy = min(cy, 1); iy <= max(cy, 1); iy++)
			for (int iz = min(cz, 1); iz <= max(cz, 1); iz++)
			{
				if (ix == 1 && iy == 1 && iz == 1) continue;

				const int bufId = (iz*3 + iy)*3 + ix;
				validHalos[current] = bufId;
				haloOffset[current] = atomicAdd(blockSum + bufId, start_size.y);
				current++;
			}

	__syncthreads();

	if (tid < 27 && blockSum[tid] > 0)
		blockSum[tid] = atomicAdd(counts + tid, blockSum[tid]);

	__syncthreads();

#pragma unroll 3
	for (int i=0; i<current; i++)
	{
		const int bufId = validHalos[i];
		const int myid  = blockSum[bufId] + haloOffset[i];

		const int ix = bufId % 3;
		const int iy = (bufId / 3) % 3;
		const int iz = bufId / 9;
		const float4 shift{ cinfo.domainSize.x*(ix-1),
							cinfo.domainSize.y*(iy-1),
							cinfo.domainSize.z*(iz-1), 0.0f };

#pragma unroll 2
		for (int i = 0; i < start_size.y; i++)
		{
			const int dstInd = 2*(myid         + i);
			const int srcInd = 2*(start_size.x + i);

			float4 tmp1 = coosvels[srcInd] - shift;
			float4 tmp2 = coosvels[srcInd+1];

			float4* addr = (float4*)dests  [bufId];
			addr[dstInd + 0] = tmp1;
			addr[dstInd + 1] = tmp2;
		}
	}
}


__global__ void getObjectHalos(const float4* __restrict__ coosvels, const ObjectVector::Properties* props, const int nObj, const int objSize,
		const int* objParticleIds, const float3 domainSize, const float rc,
		const int64_t dests[27], int counts[27], int* haloParticleIds)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	const int sh  = tid % 2;

	if (objId >= nObj) return;

	int nHalos = 0;
	short validHalos[7];

	// Find to which halos this object should go
	auto prop = props[objId];
	int cx = 1, cy = 1, cz = 1;

	if (prop.low.x  < -0.5*domainSize.x + rc) cx = 0;
	if (prop.low.y  < -0.5*domainSize.y + rc) cy = 0;
	if (prop.low.z  < -0.5*domainSize.z + rc) cz = 0;

	if (prop.high.x >  0.5*domainSize.x - rc) cx = 2;
	if (prop.high.y >  0.5*domainSize.y - rc) cy = 2;
	if (prop.high.z >  0.5*domainSize.z - rc) cz = 2;

//	if (tid == 0) printf("Obj %d : [%f %f %f] -- [%f %f %f]\n", objId,
//			prop.low.x, prop.low.y, prop.low.z, prop.high.x, prop.high.y, prop.high.z);

	for (int ix = min(cx, 1); ix <= max(cx, 1); ix++)
		for (int iy = min(cy, 1); iy <= max(cy, 1); iy++)
			for (int iz = min(cz, 1); iz <= max(cz, 1); iz++)
			{
				if (ix == 1 && iy == 1 && iz == 1) continue;
				const int bufId = (iz*3 + iy)*3 + ix;
				validHalos[nHalos] = bufId;
				nHalos++;
			}

	// Copy particles to each halo
	// TODO: maybe other loop order?
	__shared__ int shDstStart;
	const int srcStart = objId * objSize;
	for (int i=0; i<nHalos; i++)
	{
		const int bufId = validHalos[i];

		const int ix = bufId % 3;
		const int iy = (bufId / 3) % 3;
		const int iz = bufId / 9;
		const float4 shift{ domainSize.x*(ix-1),
							domainSize.y*(iy-1),
							domainSize.z*(iz-1), 0.0f };

		__syncthreads();
		if (tid == 0)
			shDstStart = atomicAdd(counts + bufId, objSize);
		__syncthreads();

		float4* dstAddr = (float4*) (dests[bufId]) + 2*shDstStart;

		for (int pid = tid/2; pid < objSize; pid += blockDim.x/2)
		{
			const int srcId = objParticleIds[srcStart + pid];
			float4 data = coosvels[2*srcId + sh];

			// Remember your origin, little particle!
			if (sh == 1)
				data.w = __int_as_float(pid);

			if (sh == 0)
				data -= shift;

			dstAddr[2*pid + sh] = data;
		}
	}
}

__global__ void addHaloForces(const float4* haloForces, const float4* halo, float4* forces, int n)
{
	const int srcId = blockIdx.x*blockDim.x + threadIdx.x;
	if (srcId >= n) return;

	const int dstId = __float_as_int(halo[2*srcId].w);
	const float4 frc = readNoCache(haloForces + srcId);
	forces[dstId] += frc;
}


HaloHelper::HaloHelper(ParticleVector* pv, const int sizes[3])
{
	CUDA_Check( cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0) );

	sendAddrs  .pushStream(stream);
	counts     .pushStream(stream);

	sendAddrs  .resize(27);
	counts     .resize(27);
	recvOffsets.resize(28);

	auto addrsPtr = sendAddrs.hostPtr();
	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

		int c = std::abs(d[0]) + std::abs(d[1]) + std::abs(d[2]);
		if (c > 0)
		{
			sendBufs[i].pushStream(stream);
			recvBufs[i].pushStream(stream);

			sendBufs[i].resize( sizes[c-1]*sizeof(Particle) );
			recvBufs[i].resize( sizes[c-1]*sizeof(Particle) );
			addrsPtr[i] = sendBufs[i].devPtr();
		}
	}
	// implicit synchro
	sendAddrs.uploadToDevice();
}

HaloExchanger::HaloExchanger(MPI_Comm& comm, cudaStream_t defStream) :
		nActiveNeighbours(26), defStream(defStream)
{
	MPI_Check( MPI_Comm_dup(comm, &haloComm) );

	int dims[3], periods[3], coords[3];
	MPI_Check( MPI_Cart_get (haloComm, 3, dims, periods, coords) );
	MPI_Check( MPI_Comm_rank(haloComm, &myrank));

	MPI_Check( MPI_Type_contiguous(sizeof(Particle), MPI_BYTE, &mpiPartType) );
	MPI_Check( MPI_Type_commit(&mpiPartType) );

	int active = 0;
	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

		int coordsNeigh[3];
		for(int c = 0; c < 3; ++c)
			coordsNeigh[c] = coords[c] + d[c];

		MPI_Check( MPI_Cart_rank(haloComm, coordsNeigh, dir2rank + i) );
		if (dir2rank[i] >= 0 && i != 13)
			compactedDirs[active++] = i;
	}
}

void HaloExchanger::attach(ParticleVector* pv, CellList* cl)
{
	particlesAndCells.push_back({pv, cl});

	const double ndens = (double)pv->np / (cl->ncells.x * cl->ncells.y * cl->ncells.z * cl->rc*cl->rc*cl->rc);
	const int maxdim = std::max({cl->domainSize.x, cl->domainSize.y, cl->domainSize.z});

	// Sizes of buffers. 0 is side, 1 is edge, 2 is corner
	const int sizes[3] = { (int)(4*ndens * maxdim*maxdim + 128), (int)(4*ndens * maxdim + 128), (int)(4*ndens + 128) };

	HaloHelper* helper = new HaloHelper(pv, sizes);
	pv->halo.pushStream(helper->stream);
	helpers.push_back(helper);
}



void HaloExchanger::attach(ObjectVector* ov, float rc)
{
	objectsAndRCs.push_back({ov, rc});

	const int maxdim = std::max({ov->domainLength.x, ov->domainLength.y, ov->domainLength.z});
	const float ndens = (double)ov->np / (ov->domainLength.x * ov->domainLength.y * ov->domainLength.z);

	const int sizes[3] = { (int)(4*ndens * maxdim*maxdim + 10*ov->objSize),
						   (int)(4*ndens * maxdim + 10*ov->objSize),
						   (int)(4*ndens + 10*ov->objSize) };


	HaloHelper* helper = new HaloHelper(ov, sizes);
	ov->halo.pushStream(helper->stream);
	ov->haloForces.pushStream(helper->stream);
	objectHelpers.push_back(helper);
}

void HaloExchanger::init()
{
	// Determine halos
	for (int i=0; i<particlesAndCells.size(); i++)
		_prepareHalos(particlesAndCells[i].first, particlesAndCells[i].second, helpers[i]);

	for (int i=0; i<objectsAndRCs.size(); i++)
		_prepareObjectHalos(objectsAndRCs[i].first, objectsAndRCs[i].second, objectHelpers[i]);

	CUDA_Check( cudaStreamSynchronize(defStream) );
}

void HaloExchanger::finalize()
{
	// Send, receive, upload to the device and sync
	for (int i=0; i<particlesAndCells.size(); i++)
		exchange(particlesAndCells[i].first->name, helpers[i], sizeof(Particle));

	for (int i=0; i<objectsAndRCs.size(); i++)
		exchange(objectsAndRCs[i].first->name, objectHelpers[i], sizeof(Particle));

	for (int i=0; i<particlesAndCells.size(); i++)
		uploadHalos(particlesAndCells[i].first, helpers[i]);

	for (int i=0; i<objectsAndRCs.size(); i++)
		uploadHalos(objectsAndRCs[i].first, objectHelpers[i]);

	for (auto& helper : helpers)
		CUDA_Check( cudaStreamSynchronize(helper->stream) );

	for (auto& objectHelper : objectHelpers)
		CUDA_Check( cudaStreamSynchronize(objectHelper->stream) );
}

void HaloExchanger::exchangeForces()
{
	for (int i=0; i<objectsAndRCs.size(); i++)
		prepareForces(objectsAndRCs[i].first, objectHelpers[i]);

	for (int i=0; i<objectsAndRCs.size(); i++)
		exchange(objectsAndRCs[i].first->name, objectHelpers[i], sizeof(Force));

	for (int i=0; i<objectsAndRCs.size(); i++)
		uploadForces(objectsAndRCs[i].first, objectHelpers[i]);

	for (auto& helper : objectHelpers)
		CUDA_Check( cudaStreamSynchronize(helper->stream) );
}


void HaloExchanger::exchange(std::string pvName, HaloHelper* helper, int typeSize)
{

	auto tagByName = [] (std::string name) {
		static std::hash<std::string> nameHash;
		return (int)( nameHash(name) % 414243 );
	};

	// Post receives
	helper->requests.clear();
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			MPI_Request req;

			// Invert the direction index
			const int cx = -( i%3 - 1 ) + 1;
			const int cy = -( (i/3)%3 - 1 ) + 1;
			const int cz = -( i/9 - 1 ) + 1;

			const int invDirCode = (cz*3 + cy)*3 + cx;
			const int tag = 27 * tagByName(pvName) + invDirCode;

			MPI_Check( MPI_Irecv(helper->recvBufs[i].hostPtr(), helper->recvBufs[i].size(), MPI_BYTE, dir2rank[i], tag, haloComm, &req) );
			helper->requests.push_back(req);
		}

	// Prepare message sizes and send
	helper->counts.downloadFromDevice();
	auto cntPtr = helper->counts.hostPtr();
	for (int i=0; i<27; i++)
		if (i != 13)
		{
			if (cntPtr[i]*typeSize > helper->sendBufs[i].size())
				die("Preallocated halo buffer %d for %s too small: size %d bytes, but need %d bytes",
						i, pvName.c_str(), helper->sendBufs[i].size(), cntPtr[i]*typeSize);

			if (cntPtr[i] > 0)
				CUDA_Check( cudaMemcpyAsync(helper->sendBufs[i].hostPtr(), helper->sendBufs[i].devPtr(),
						cntPtr[i]*typeSize, cudaMemcpyDeviceToHost, helper->stream) );
		}
	CUDA_Check( cudaStreamSynchronize(helper->stream) );

	MPI_Request req;
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			debug3("Sending %s halo to rank %d in dircode %d [%2d %2d %2d], %d particles", pvName.c_str(), dir2rank[i], i, i%3 - 1, (i/3)%3 - 1, i/9 - 1, cntPtr[i]);
			const int tag = 27 * tagByName(pvName) + i;
			MPI_Check( MPI_Isend(helper->sendBufs[i].hostPtr(), cntPtr[i]*typeSize, MPI_BYTE, dir2rank[i], tag, haloComm, &req) );
			MPI_Check( MPI_Request_free(&req) );
		}

	// Wait until messages are arrived
	const int nMessages = helper->requests.size();
	std::vector<MPI_Status> statuses(nMessages);
	MPI_Check( MPI_Waitall(nMessages, helper->requests.data(), statuses.data()) );

	// Excl scan of message sizes to know where we will upload them
	int totalRecvd = 0;
	std::fill(helper->recvOffsets.begin(), helper->recvOffsets.end(), std::numeric_limits<int>::max());
	for (int i=0; i<nMessages; i++)
	{
		helper->recvOffsets[compactedDirs[i]] = totalRecvd;

		int msize;
		MPI_Check( MPI_Get_count(&statuses[i], MPI_BYTE, &msize) );
		msize /= typeSize;
		totalRecvd += msize;

		debug3("Receiving %s halo from rank %d, %d particles", pvName.c_str(), dir2rank[compactedDirs[i]], msize);
	}

	// Fill the holes in the offsets
	helper->recvOffsets[27] = totalRecvd;
	for (int i=0; i<27; i++)
		helper->recvOffsets[i] = std::min(helper->recvOffsets[i+1], helper->recvOffsets[i]);
}

void HaloExchanger::_prepareHalos(ParticleVector* pv, CellList* cl, HaloHelper* helper)
{
	debug2("Preparing %s halo on the device", pv->name.c_str());

	helper->counts.pushStream(defStream);
	helper->counts.clearDevice();
	helper->counts.popStream();

	const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
	const int nthreads = 32;
	if (pv->np > 0)
		getHalos<<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 6, 1),  dim3(nthreads, 1, 1), 0, defStream >>>
				((float4*)pv->coosvels.devPtr(), cl->cellInfo(), cl->cellsStartSize.devPtr(), (int64_t*)helper->sendAddrs.devPtr(), helper->counts.devPtr());
}


void HaloExchanger::_prepareObjectHalos(ObjectVector* ov, float rc, HaloHelper* helper)
{
	debug2("Preparing %s halo on the device", ov->name.c_str());

	helper->counts.pushStream(defStream);
	helper->counts.clearDevice();
	helper->counts.popStream();

	const int nthreads = 128;
	if (ov->nObjects > 0)
		getObjectHalos <<< ov->nObjects, nthreads, 0, defStream >>>
				((float4*)ov->coosvels.devPtr(), ov->properties.devPtr(), ov->nObjects, ov->objSize, ov->particles2objIds.devPtr(), ov->domainLength, rc,
				 (int64_t*)helper->sendAddrs.devPtr(), helper->counts.devPtr(), ov->haloIds.devPtr());
}

void HaloExchanger::uploadHalos(ParticleVector* pv, HaloHelper* helper)
{
	pv->halo.resize(helper->recvOffsets[27], resizeAnew);

	for (int i=0; i < 27; i++)
	{
		const int msize = helper->recvOffsets[i+1] - helper->recvOffsets[i];
		if (msize > 0)
			CUDA_Check( cudaMemcpyAsync(pv->halo.devPtr() + helper->recvOffsets[i], helper->recvBufs[i].hostPtr(),
					msize*sizeof(Particle), cudaMemcpyHostToDevice, helper->stream) );
	}
}

void HaloExchanger::prepareForces(ObjectVector* ov, HaloHelper* helper)
{
	debug2("Preparing %s halo on the device", ov->name.c_str());

	for (int i=0; i<27; i++)
	{
		helper->counts[i] = helper->recvOffsets[i+1] - helper->recvOffsets[i];
		if (helper->counts[i] > 0)
			CUDA_Check( cudaMemcpyAsync(ov->haloForces.devPtr() + helper->recvOffsets[i], helper->sendBufs[i].hostPtr(),
					helper->counts[i]*sizeof(Force), cudaMemcpyHostToDevice, helper->stream) );
	}

	// implicit synchronization here
	helper->counts.uploadToDevice();
}

void HaloExchanger::uploadForces(ObjectVector* ov, HaloHelper* helper)
{
	for (int i=0; i < helper->recvOffsets.size(); i++)
	{
		const int msize = helper->recvOffsets[i+1] - helper->recvOffsets[i];

		if (msize > 0)
			CUDA_Check( cudaMemcpyAsync(ov->haloForces.devPtr() + helper->recvOffsets[i], helper->recvBufs[compactedDirs[i]].hostPtr(),
					msize*sizeof(Force), cudaMemcpyHostToDevice, helper->stream) );
	}

	const int np = helper->recvOffsets[27];
	addHaloForces<<< (np+127)/128, 128, 0, helper->stream >>> ( (float4*)ov->haloForces.devPtr(), (float4*)ov->halo.devPtr(), (float4*)ov->forces.devPtr(), np);
}


