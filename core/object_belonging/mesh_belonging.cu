#include "mesh_belonging.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/celllist.h>

#include <core/rigid_kernels/quaternion.h>
#include <core/rigid_kernels/rigid_motion.h>

const float tolerance = 5e-6f;

__device__ __forceinline__ float tetrahedronVolume(float3 a, float3 b, float3 c, float3 d)
{
	// https://en.wikipedia.org/wiki/Tetrahedron#Volume
	return fabs( (1.0f/6.0f) * dot(a-d, cross(b-d, c-d)) );
}


__device__ __forceinline__ int particleInsideTetrahedron(float3 r, float3 v0, float3 v1, float3 v2, float3 v3)
{
	float V = tetrahedronVolume(v0, v1, v2, v3);

	if (fabs(V) < tolerance) return 0;

	float V0 = tetrahedronVolume(r,  v1, v2, v3);
	float V1 = tetrahedronVolume(v0,  r, v2, v3);
	float V2 = tetrahedronVolume(v0, v1,  r, v3);
	float V3 = tetrahedronVolume(v0, v1, v2,  r);

	if (V0 == 0.0f || V1 == 0.0f || V2 == 0.0f || V3 == 0.0f)
		return 1;

	if (fabs(V - V0+V1+V2+V3) < tolerance) return 2;

	return 0;
}


/**
 * One warp works on one particle
 */
__device__ BelongingTags oneParticleInsideMesh(float3 r, int objId, const float3 com, const MeshView mesh)
{
	int counter = 0;

	// Work in obj reference frame for simplicity
	r = r - com;

	for (int i = __laneid(); i < mesh.nvertices; i += warpSize)
	{
		float3 v0 = make_float3(0);

		int3 trid = mesh.triangles[i];

		float3 v1 = Particle(mesh.vertices, objId*mesh.nvertices + trid.x).r - com;
		float3 v2 = Particle(mesh.vertices, objId*mesh.nvertices + trid.y).r - com;
		float3 v3 = Particle(mesh.vertices, objId*mesh.nvertices + trid.z).r - com;

		// If the particle is very close to the boundary
		// return immediately
		if ( dot(r-v1, cross(v2-v1, v3-v1)) < tolerance )
			return BelongingTags::Boundary;

		// += 2 if inside
		// += 1 if exactly on a side
		counter += particleInsideTetrahedron(r, v0, v1, v2, v3);
	}

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

	const int widInObj = wid % WARPS_PER_OBJ;

	if (objId >= view.nObjects) return;

	const int3 cidLow  = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].low  - 1.5f);
	const int3 cidHigh = cinfo.getCellIdAlongAxes(view.comAndExtents[objId].high + 2.5f);

	const int3 span = cidHigh - cidLow + make_int3(1,1,1);
	const int totCells = span.x * span.y * span.z;

	for (int i=widInObj; i<totCells; i+=WARPS_PER_OBJ)
	{
		const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;
		const int  cid = cinfo.encode(cid3);
		if (cid >= cinfo.totcells) continue;

		int pstart = cinfo.cellStarts[cid];
		int pend   = cinfo.cellStarts[cid+1];

		for (int pid = pstart; pid < pend; pid++)
		{
			const Particle p(cinfo.particles, pid);

			tags[pid] = oneParticleInsideMesh(p.r, objId, view.comAndExtents[objId].com, mesh);
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

	SAFE_KERNEL_LAUNCH(
			insideMesh<warpsPerObject>,
			warpsPerObject*view.nObjects, nthreads, 0, stream,
			view, meshView, cl->cellInfo(), tags.devPtr());

	// Halo
	lov = ov->halo();       // Note ->halo() here
	view = OVview(ov, lov);
	vertices = lov->getMeshVertices(stream);
	meshView = MeshView(ov->mesh, lov->getMeshVertices(stream));

	SAFE_KERNEL_LAUNCH(
			insideMesh<warpsPerObject>,
			warpsPerObject*view.nObjects, nthreads, 0, stream,
			view, meshView, cl->cellInfo(), tags.devPtr());
}



