#include "inside_checker.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/celllist.h>

#include <core/rigid_kernels/quaternion.h>


__global__ void copyLeftRight(const Particle* srcs, const int n,
		const int* tags, Particle* lefts, Particle* rights, int* nLeft, int* nRight)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= gid) return;

	const int tag = tags[gid];
	const Particle p = srcs[gid];

	if (tag == -1)
	{
		int dstId = atomicAggInc(nLeft);
		if (lefts)
			lefts[dstId] = p;
	}

	if (tag >= 0)
	{
		int dstId = atomicAggInc(nRight);
		if (rights)
			rights[dstId] = p;
	}
}

__device__ inline float ellipsoidF(const float3 r, const float3 invAxes)
{
	return sqr(r.x * invAxes.x) + sqr(r.y * invAxes.y) + sqr(r.z * invAxes.z) - 1.0f;
}

__global__ void insideEllipsoid(
		LocalObjectVector::COMandExtent* props, LocalRigidEllipsoidObjectVector::RigidMotion* motions, float3 invAxes, int nObj,
		const Particle* particles, const uint* cellsStartSize, CellListInfo cinfo,
		int* tags, int* nInside, int* nOutside)
{
	const float tolerance = 5e-6f;

	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	if (objId >= nObj) return;

	const int3 cidLow  = cinfo.getCellIdAlongAxis(props[objId].low  - 0.5f);
	const int3 cidHigh = cinfo.getCellIdAlongAxis(props[objId].high + 1.5f);

	const int3 span = cidHigh - cidLow + make_int3(1,1,1);
	const int totCells = span.x * span.y * span.z;

	for (int i=tid; i<totCells + blockDim.x-1; i+=blockDim.x)
	{
		const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
		int2 start_size = cinfo.decodeStartSize(cellsStartSize[validCells[threadIdx.x]]);

		for (int pid = start_size.x; pid < start_size.x + start_size.y; pid++)
		{
			const Particle p = particles[pid];
			float3 coo = rotate(p.r - motions[objId].r, invQ(motions[objId].q));

			float v = ellipsoidF(coo, invAxes);

			if (fabs(v) <= tolerance)
			{
				tags[pid] = -2;
			}
			else if (v <= tolerance)
			{
				atomicAggInc(nInside);
				tags[pid] = objId;
			}
			else
			{
				atomicAggInc(nOutside);
				tags[pid] = -1;
			}
		}
	}
}


static void InsideChecker::splitByTags(ParticleVector* src, PinnedBuffer<int>& tags, int nInside, int nOutside,
		ParticleVector* pvIn, ParticleVector* pvOut, cudaStream_t stream)
{
	PinnedBuffer<int> nLefts(1), nRights(1);
	nLefts.clear(stream);
	nRights.clear(stream);

	if (pvIn != nullptr)  pvIn-> local()->resize(nInside, stream);
	if (pvOut != nullptr) pvOut->local()->resize(nInside, stream);

	copyLeftRight <<< getNblocks(src->local()->size(), 128), 128, 0, stream >>> (
			src->local()->coosvels.devPtr(), src->local()->size(),
			tags.devPtr(),
			pvIn ?  pvIn-> local()->coosvels.devPtr() : nullptr,
			pvOut ? pvOut->local()->coosvels.devPtr() : nullptr,
			nLefts.devPtr(), nRights.devPtr() );
}


void EllipsoidInsideChecker::tagInner(ParticleVector* pv, CellList* cl, PinnedBuffer<int>& tags, int& nInside, int& nOutside, cudaStream_t stream)
{
	nIn.clear(stream);
	nOut.clear(stream);

	int nthreads = 512;

	auto activeROV = rov->local();
	insideEllipsoid<<< activeROV->nObjects, nthreads, 2*nthreads*sizeof(int), stream >>> (
			activeROV->comAndExtents.devPtr(), activeROV->motions.devPtr(), 1.0f / rov->axes, activeROV->nObjects,
			pv->local()->coosvels.devPtr(), cl->cellsStartSize.devPtr(), cl->cellInfo(),
			tags.devPtr(), nIn.devPtr(), nOut.devPtr());

	activeROV = rov->halo();
	insideEllipsoid<<< activeROV->nObjects, nthreads, 2*nthreads*sizeof(int), stream >>> (
			activeROV->comAndExtents.devPtr(), activeROV->motions.devPtr(), 1.0f / rov->axes, activeROV->nObjects,
			pv->local()->coosvels.devPtr(), cl->cellsStartSize.devPtr(), cl->cellInfo(),
			tags.devPtr(), nIn.devPtr(), nOut.devPtr());

	nIn. downloadFromDevice(stream);
	nOut.downloadFromDevice(stream);

	nInside = nIn[0];
	nOutside = nOut[0];
}
