#include "mesh_belonging.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/celllist.h>

#include <core/rigid_kernels/quaternion.h>
#include <core/rigid_kernels/rigid_motion.h>

const float tolerance = 1e-6f;

__device__ __forceinline__ float tetrahedronVolume(float3 a, float3 b, float3 c, float3 d)
{
	// https://en.wikipedia.org/wiki/Tetrahedron#Volume
	return fabs( (1.0f/6.0f) * dot(a-d, cross(b-d, c-d)) );
}


__device__ __forceinline__ int particleInsideTetrahedron(float3 r, float3 v0, float3 v1, float3 v2, float3 v3)
{
	float V = tetrahedronVolume(v0, v1, v2, v3);

	float V0 = tetrahedronVolume(r,  v1, v2, v3);
	float V1 = tetrahedronVolume(v0,  r, v2, v3);
	float V2 = tetrahedronVolume(v0, v1,  r, v3);
	float V3 = tetrahedronVolume(v0, v1, v2,  r);

	if (fabs(V - (V0+V1+V2+V3)) > tolerance) return 0;

//	printf("%e  %e %e %e %e  %e\n", V, V0, V1, V2, V3, V0+V1+V2+V3);

	// Volumes sum up, but one of them is 0. Therefore particle is exactly on one side
	// Another tetrahedron with the same side will also contribute 1
	const float vtol = 0.1f*tolerance;
	if (V0 < vtol || V1 < vtol || V2 < vtol || V3 < vtol) return 1;

	return 2;
}


/**
 * One warp works on one particle
 */
__device__ BelongingTags oneParticleInsideMesh(int pid, float3 r, int objId, const float3 com, const MeshView mesh)
{
	int counter = 0;

	// Work in obj reference frame for simplicity
	r = r - com;

	float3 tot = make_float3(0);

	for (int i = __laneid(); i < mesh.ntriangles; i += warpSize)
	{
		float3 v0 = make_float3(0);

		int3 trid = mesh.triangles[i];

		float3 v1 = Particle(mesh.vertices, objId*mesh.nvertices + trid.x).r - com;
		float3 v2 = Particle(mesh.vertices, objId*mesh.nvertices + trid.y).r - com;
		float3 v3 = Particle(mesh.vertices, objId*mesh.nvertices + trid.z).r - com;

		// If the particle is very close to the boundary
		// return immediately
		if ( fabs( dot(r-v1, normalize(cross(v2-v1, v3-v1))) ) < tolerance )
			return BelongingTags::Boundary;

		// += 2 if inside
		// += 1 if exactly on a side
		counter += particleInsideTetrahedron(r, v0, v1, v2, v3);
	}

	counter = warpReduce(counter, [] (int a, int b) { return a+b; });

	// Incorrect result. Disregard the guy just in case
	if (counter % 2 != 0) return BelongingTags::Boundary;


	// Inside even number of tetrahedra => outside of object
	if ( (counter/2) % 2 == 0 ) return BelongingTags::Outside;

	// Inside odd number of tetrahedra => inside object
	if ( (counter/2) % 2 != 0 ) return BelongingTags::Inside;

	// Shut up compiler warning
	return BelongingTags::Boundary;
}

/**
 * OVview view is only used to provide # of objects and extent information
 * Actual data is in mesh.vertices
 * cinfo is the cell-list sync'd with the target ParticleVector data
 */
template<int WARPS_PER_OBJ>
__global__ void insideMesh(const OVview view, const MeshView mesh, CellListInfo cinfo, BelongingTags* tags)
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int wid = gid / warpSize;
	const int objId = wid / WARPS_PER_OBJ;

	const int locWid = wid % WARPS_PER_OBJ;

	if (objId >= view.nObjects) return;

	const int3 cidLow  = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].low  - 0.8f);
	const int3 cidHigh = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].high + 1.0f);

	const int3 span = cidHigh - cidLow + make_int3(1,1,1);
	const int totCells = span.x * span.y * span.z;

	for (int i=locWid; i<totCells; i+=WARPS_PER_OBJ)
	{
		const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
		const int  cid = cinfo.encode(cid3);
		if (cid < 0 || cid >= cinfo.totcells) continue;

		int pstart = cinfo.cellStarts[cid];
		int pend   = cinfo.cellStarts[cid+1];

		for (int pid = pstart; pid < pend; pid++)
		{
			const Particle p(cinfo.particles, pid);

			auto tag = oneParticleInsideMesh(pid, p.r, objId, view.comAndExtents[objId].com, mesh);
			if (__laneid() == 0) tags[pid] = tag;
		}
	}
}


void MeshBelongingChecker::tagInner(ParticleVector* pv, CellList* cl, cudaStream_t stream)
{
	int nthreads = 128;

	tags.resize_anew(pv->local()->size());
	tags.clearDevice(stream);

	const int warpsPerObject = 32;

	// Local
	auto lov = ov->local();
	auto view = OVview(ov, lov);
	auto vertices = lov->getMeshVertices(stream);
	auto meshView = MeshView(ov->mesh, lov->getMeshVertices(stream));

	debug("Computing inside/outside tags (against mesh) for %d local objects '%s' and %d '%s' particles",
			view.nObjects, ov->name.c_str(), pv->local()->size(), pv->name.c_str());

	SAFE_KERNEL_LAUNCH(
			insideMesh<warpsPerObject>,
			getNblocks(warpsPerObject*32*view.nObjects, nthreads), nthreads, 0, stream,
			view, meshView, cl->cellInfo(), tags.devPtr());

	// Halo
	lov = ov->halo();       // Note ->halo() here
	view = OVview(ov, lov);
	vertices = lov->getMeshVertices(stream);
	meshView = MeshView(ov->mesh, lov->getMeshVertices(stream));

	debug("Computing inside/outside tags (against mesh) for %d halo objects '%s' and %d '%s' particles",
			view.nObjects, ov->name.c_str(), pv->local()->size(), pv->name.c_str());

	SAFE_KERNEL_LAUNCH(
			insideMesh<warpsPerObject>,
			getNblocks(warpsPerObject*32*view.nObjects, nthreads), nthreads, 0, stream,
			view, meshView, cl->cellInfo(), tags.devPtr());
}



