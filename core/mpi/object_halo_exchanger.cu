#include <core/particle_vector.h>
#include <core/object_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/cuda_common.h>

#include <core/mpi/object_halo_exchanger.h>

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

__global__ void getObjectHalos(const float4* __restrict__ coosvels, const ObjectVector::COMandExtent* props, const int nObj, const int objSize,
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


void ObjectHaloExchanger::attach(ObjectVector* ov, float rc)
{
	objects.push_back(ov);
	rcs.push_back(rc);

	const int maxdim = std::max({ov->domainSize.x, ov->domainSize.y, ov->domainSize.z});
	const float ndens = (double)ov->np / (ov->domainSize.x * ov->domainSize.y * ov->domainSize.z);

	const int sizes[3] = { (int)(4*ndens * maxdim*maxdim + 10*ov->objSize),
						   (int)(4*ndens * maxdim + 10*ov->objSize),
						   (int)(4*ndens + 10*ov->objSize) };


	HaloHelper* helper = new HaloHelper(sizes, &ov->halo);
	ov->halo.pushStream(helper->stream);
	ov->haloForces.pushStream(helper->stream);
	objectHelpers.push_back(helper);
}


void ObjectHaloExchanger::exchangeForces()
{
	for (int i=0; i<objects.size(); i++)
		prepareForces(objects[i], helpers[i]);

	for (int i=0; i<objects.size(); i++)
		exchange(helpers[i], sizeof(Force));

	for (int i=0; i<objects.size(); i++)
		uploadForces(objects[i], helpers[i]);

	for (auto helper : helpers)
		CUDA_Check( cudaStreamSynchronize(helper->stream) );
}


void ObjectHaloExchanger::_prepareHalos(int id)
{
	auto ov = objects[id];
	auto rc = rcs[id];
	auto helper = helpers[id];

	debug2("Preparing %s halo on the device", ov->name.c_str());

	helper->counts.pushStream(defStream);
	helper->counts.clearDevice();
	helper->counts.popStream();

	const int nthreads = 128;
	if (ov->nObjects > 0)
		getObjectHalos <<< ov->nObjects, nthreads, 0, defStream >>>
				((float4*)ov->coosvels.devPtr(), ov->com_extent.devPtr(), ov->nObjects, ov->objSize, ov->particles2objIds.devPtr(), ov->domainSize, rc,
				 (int64_t*)helper->sendAddrs.devPtr(), helper->counts.devPtr(), ov->haloIds.devPtr());
}

void ObjectHaloExchanger::prepareForces(ObjectVector* ov, HaloHelper* helper)
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

void ObjectHaloExchanger::uploadForces(ObjectVector* ov, HaloHelper* helper)
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


