#include "ellipsoid_belonging.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/celllist.h>

#include <core/rigid_kernels/quaternion.h>


__device__ inline float ellipsoidF(const float3 r, const float3 invAxes)
{
	return sqr(r.x * invAxes.x) + sqr(r.y * invAxes.y) + sqr(r.z * invAxes.z) - 1.0f;
}

__global__ void insideEllipsoid(REOVview view, CellListInfo cinfo, int* tags)
{
	const float tolerance = 5e-6f;

	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	if (objId >= view.nObjects) return;

	const int3 cidLow  = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].low  - 1.5f);
	const int3 cidHigh = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].high + 2.5f);

	const int3 span = cidHigh - cidLow + make_int3(1,1,1);
	const int totCells = span.x * span.y * span.z;

	for (int i=tid; i<totCells; i+=blockDim.x)
	{
		const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
		const int  cid = cinfo.encode(cid3);
		if (cid >= cinfo.totcells) continue;

		int pstart = cinfo.cellStarts[cid];
		int pend   = cinfo.cellStarts[cid+1];

		for (int pid = pstart; pid < pend; pid++)
		{
			const Particle p(cinfo.particles, pid);
			float3 coo = rotate(p.r - view.motions[objId].r, invQ(view.motions[objId].q));

			float v = ellipsoidF(coo, view.invAxes);

			if (fabs(v) <= tolerance) // boundary
				tags[pid] = -84;
			else if (v <= tolerance)  // inside
				tags[pid] = 1;
		}
	}
}


void EllipsoidBelongingChecker::tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream)
{
	int nthreads = 512;

	auto reov = dynamic_cast<RigidEllipsoidObjectVector*> (ov);
	if (reov == nullptr)
		die("Ellipsoid belonging can only be used with ellipsoid objects (%s is not)", ov->name.c_str());

	tags.resize_anew(pv->local()->size());
	tags.clearDevice(stream);

	//ov->findExtentAndCOM(stream);

	auto view = REOVview(reov, reov->local());
	SAFE_KERNEL_LAUNCH(
			insideEllipsoid,
			view.nObjects, nthreads, 0, stream,
			view, cl->cellInfo(), tags.devPtr());

	view = REOVview(reov, reov->halo());
	SAFE_KERNEL_LAUNCH(
			insideEllipsoid,
			view.nObjects, nthreads, 0, stream,
			view, cl->cellInfo(), tags.devPtr());
}



