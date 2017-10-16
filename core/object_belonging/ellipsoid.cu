#include "ellipsoid.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/celllist.h>

#include <core/rigid_kernels/quaternion.h>


__device__ inline float ellipsoidF(const float3 r, const float3 invAxes)
{
	return sqr(r.x * invAxes.x) + sqr(r.y * invAxes.y) + sqr(r.z * invAxes.z) - 1.0f;
}

__global__ void insideEllipsoid(
		REOVview view, CellListInfo cinfo,
		int* tags, int* nInside, int* nOutside)
{
	const float tolerance = 5e-6f;

	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	if (objId >= view.nObjects) return;

	const int3 cidLow  = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].low  - 0.5f);
	const int3 cidHigh = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].high + 1.5f);

	const int3 span = cidHigh - cidLow + make_int3(1,1,1);
	const int totCells = span.x * span.y * span.z;

	for (int i=tid; i<totCells + blockDim.x-1; i+=blockDim.x)
	{
		const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
		const int  cid = cinfo.encode(cid3);

		int pstart = cinfo.cellStarts[cid];
		int pend   = cinfo.cellStarts[cid+1];

		for (int pid = pstart; pid < pend; pid++)
		{
			const Particle p = (view.particles, pid);
			float3 coo = rotate(p.r - view.motions[objId].r, invQ(view.motions[objId].q));

			float v = ellipsoidF(coo, view.invAxes);

			if (fabs(v) <= tolerance)
			{
				// Boundary layer
				tags[pid] = -84;
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


void EllipsoidBelongingChecker::tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream)
{
	nInside.clearDevice(stream);
	nOutside.clearDevice(stream);

	int nthreads = 512;

	auto reov = dynamic_cast<RigidEllipsoidObjectVector*> (ov);
	if (reov == nullptr)
		die("Ellipsoid belonging can only be used with ellipsoid objects (%s is not)", ov->name.c_str());

	SAFE_KERNEL_LAUNCH(
			insideEllipsoid,
			REOVview(reov, reov->local()), cl->cellInfo(),
			tags.devPtr(), nInside.devPtr(), nOutside.devPtr() );

	activeROV = ov->halo();
	SAFE_KERNEL_LAUNCH(
			insideEllipsoid,
			REOVview(reov, reov->halo()), cl->cellInfo(),
			tags.devPtr(), nInside.devPtr(), nOutside.devPtr() );

	nInside. downloadFromDevice(stream);
	nOutside.downloadFromDevice(stream);
}



