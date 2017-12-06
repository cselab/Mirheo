#include <cassert>
#include <type_traits>

#include <core/celllist.h>
#include <core/utils/cuda_common.h>

/// Squared distance between vectors with components
/// (\p a.x, \p a.y, \p a.z) and (\p b.x, \p b.y, \p b.z)
template<typename Ta, typename Tb>
__device__ inline float distance2(const Ta a, const Tb b)
{
	auto sqr = [] (float x) { return x*x; };
	return sqr(a.x - b.x) + sqr(a.y - b.y) + sqr(a.z - b.z);
}


/**
 * Compute interactions between one destination particle and
 * all source particles in a given cell, defined by range of ids:
 *
 * \code
 * for (id = pstart; id < pend; id++)
 * F += interaction(dstP, Particle(cinfo.particles, id));
 * \endcode
 *
 * Also update forces for the source particles in process.
 *
 * Source particles may be from the same ParticleVector or from
 * different ones.
 *
 * @param pstart lower bound of id range of the particles to be worked on
 * @param pend  upper bound of id range
 * @param dstP destination particle
 * @param dstId destination particle local id, may be used to fetch extra
 *        properties associated with the given particle
 * @param dstFrc target force
 * @param cinfo cell-list data for source particles
 * @param rc2 squared interaction cut-off distance
 * @param interaction interaction implementation, see computeSelfInteractions()
 *
 * @tparam NeedDstAcc whether to update \p dstFrc or not. One out of
 * \p NeedDstAcc or \p NeedSrcAcc should be true.
 * @tparam NeedSrcAcc whether to update forces for source particles.
 * One out of \p NeedDstAcc or \p NeedSrcAcc should be true.
 * @tparam Self true if we're computing self interactions, meaning
 * that destination particle is one of the source particles.
 * In that case only half of the interactions contribute to the
 * forces, such that either p1 \<-\> p2 or p2 \<-\> p1 is ignored
 * based on particle ids
 */
template<bool NeedDstAcc, bool NeedSrcAcc, bool Self, typename Interaction>
__device__ inline void computeCell(
		int pstart, int pend,
		Particle dstP, int dstId, float3& dstFrc,
		CellListInfo cinfo,
		float rc2, Interaction& interaction)
{
	for (int srcId = pstart; srcId < pend; srcId++)
	{
		Particle srcP;
		srcP.readCoordinate(cinfo.particles, srcId);

		bool interacting = distance2(srcP.r, dstP.r) < rc2;

		if (Self)
			if (dstId <= srcId) interacting = false;

		if (interacting)
		{
			srcP.readVelocity(cinfo.particles, srcId);

			float3 frc = interaction(dstP, dstId, srcP, srcId);

			if (NeedDstAcc)
				dstFrc += frc;

			if (NeedSrcAcc)
				atomicAdd(cinfo.forces + srcId, -frc);
		}
	}
}

/**
 * Compute interactions within a single ParticleVector.
 *
 * Mapping is one thread per particle. The thread will traverse half
 * of the neighbouring cells and compute all the interactions between
 * original destination particle and all the particles in the cells.
 *
 * @param np number of particles
 * @param cinfo cell-list data
 * @param rc2 squared cut-off distance
 * @param interaction is a \c \_\_device\_\_ callable that computes
 *        the force between two particles. It has to have the following
 *        signature:
 *        \code float3 interaction(const Particle dst, int dstId, const Particle src, int srcId) \endcode
 *        The return value is the force acting on the first particle.
 *        The second one experiences the opposite force.
 */
template<typename Interaction>
//__launch_bounds__(128, 16)
__global__ void computeSelfInteractions(
		const int np, CellListInfo cinfo,
		const float rc2, Interaction interaction)
{
	const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
	if (dstId >= np) return;

	const Particle dstP(cinfo.particles, dstId);
	float3 dstFrc = make_float3(0.0f);

	const int3 cell0 = cinfo.getCellIdAlongAxes(dstP.r);

	for (int cellZ = cell0.z-1; cellZ <= cell0.z+1; cellZ++)
		for (int cellY = cell0.y-1; cellY <= cell0.y; cellY++)
			{
				if ( !(cellY >= 0 && cellY < cinfo.ncells.y && cellZ >= 0 && cellZ < cinfo.ncells.z) ) continue;
				if (cellY == cell0.y && cellZ > cell0.z) continue;

				const int midCellId = cinfo.encode(cell0.x, cellY, cellZ);
				int rowStart  = max(midCellId-1, 0);
				int rowEnd    = min(midCellId+2, cinfo.totcells);

				if ( cellY == cell0.y && cellZ == cell0.z ) rowEnd = midCellId + 1; // this row is already partly covered

				const int pstart = cinfo.cellStarts[rowStart];
				const int pend   = cinfo.cellStarts[rowEnd];

				if (cellY == cell0.y && cellZ == cell0.z)
					computeCell<true, true, true>  (pstart, pend, dstP, dstId, dstFrc, cinfo, rc2, interaction);
				else
					computeCell<true, true, false> (pstart, pend, dstP, dstId, dstFrc, cinfo, rc2, interaction);
			}

	atomicAdd(cinfo.forces + dstId, dstFrc);
}

/**
 * Compute interactions between particle of two different ParticleVector.
 *
 * Mapping is one thread per particle. The thread will traverse all
 * of the neighbouring cells and compute all the interactions between
 * original destination particle and all the particles of the second
 * kind residing in the cells.
 *
 * @param dstView view of the destination particles. They are accessed
 *        by the threads in a completely coalesced manner, no cell-list
 *        is needed for them.
 * @param srcCinfo cell-list data for the source particles
 * @param rc2 squared cut-off distance
 * @param interaction is a \c \_\_device\_\_ callable that computes
 *        the force between two particles. It has to have the following
 *        signature:
 *        \code float3 interaction(const Particle dst, int dstId, const Particle src, int srcId) \endcode
 *        The return value is the force acting on the first particle.
 *        The second one experiences the opposite force.
 *
 * @tparam NeedDstAcc if true, compute forces for destination particles.
 *         One out of \p NeedDstAcc or \p NeedSrcAcc should be true.
 * @tparam NeedSrcAcc if true, compute forces for source particles.
 *         One out of \p NeedDstAcc or \p NeedSrcAcc should be true.
 * @tparam Variant performance related parameter. \e true is better for
 * densely mixed stuff, \e false is better for halo
 */
template<bool NeedDstAcc, bool NeedSrcAcc, bool Variant, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions_1tpp(
		PVview dstView, CellListInfo srcCinfo,
		const float rc2, Interaction interaction)
{
	static_assert(NeedDstAcc || NeedSrcAcc, "External interactions should return at least some accelerations");

	const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
	if (dstId >= dstView.size) return;

	const Particle dstP(
			readNoCache(dstView.particles+2*dstId),
			readNoCache(dstView.particles+2*dstId+1) );

	float3 dstFrc = make_float3(0.0f);

	const int3 cell0 = srcCinfo.getCellIdAlongAxes<false>(dstP.r);

	for (int cellZ = cell0.z-1; cellZ <= cell0.z+1; cellZ++)
		for (int cellY = cell0.y-1; cellY <= cell0.y+1; cellY++)
			if (Variant)
			{
				if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) continue;

				const int midCellId = srcCinfo.encode(cell0.x, cellY, cellZ);
				int rowStart  = max(midCellId-1, 0);
				int rowEnd    = min(midCellId+2, srcCinfo.totcells);

				const int pstart = srcCinfo.cellStarts[rowStart];
				const int pend   = srcCinfo.cellStarts[rowEnd];

				computeCell<NeedDstAcc, NeedSrcAcc, false> (pstart, pend, dstP, dstId, dstFrc, srcCinfo, rc2, interaction);
			}
			else
			{
				if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) continue;

				for (int cellX = max(cell0.x-1, 0); cellX <= min(cell0.x+1, srcCinfo.ncells.x-1); cellX++)
				{
					const int cid = srcCinfo.encode(cellX, cellY, cellZ);
					const int pstart = srcCinfo.cellStarts[cid];
					const int pend   = srcCinfo.cellStarts[cid+1];

					computeCell<NeedDstAcc, NeedSrcAcc, false> (pstart, pend, dstP, dstId, dstFrc, srcCinfo, rc2, interaction);
				}
			}

	if (NeedDstAcc)
		atomicAdd(dstView.forces + dstId, dstFrc);
}

/**
 * Compute interactions between particle of two different ParticleVector.
 *
 * Mapping is three threads per particle. The rest is similar to
 * computeExternalInteractions_1tpp()
 */
template<bool NeedDstAcc, bool NeedSrcAcc, bool Variant, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions_3tpp(
		PVview dstView, CellListInfo srcCinfo,
		const float rc2, Interaction interaction)
{
	static_assert(NeedDstAcc || NeedSrcAcc, "External interactions should return at least some accelerations");

	const int gid = blockIdx.x*blockDim.x + threadIdx.x;

	const int dstId = gid / 3;
	const int dircode = gid % 3 - 1;

	if (dstId >= dstView.size) return;

	const Particle dstP(
			readNoCache(dstView.particles+2*dstId),
			readNoCache(dstView.particles+2*dstId+1) );

	float3 dstFrc = make_float3(0.0f);

	const int3 cell0 = srcCinfo.getCellIdAlongAxes<false>(dstP.r);

	int cellZ = cell0.z + dircode;

	for (int cellY = cell0.y-1; cellY <= cell0.y+1; cellY++)
		if (Variant)
		{
			if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) continue;

			const int midCellId = srcCinfo.encode(cell0.x, cellY, cellZ);
			int rowStart  = max(midCellId-1, 0);
			int rowEnd    = min(midCellId+2, srcCinfo.totcells);

			const int pstart = srcCinfo.cellStarts[rowStart];
			const int pend   = srcCinfo.cellStarts[rowEnd];

			computeCell<NeedDstAcc, NeedSrcAcc, false> (pstart, pend, dstP, dstId, dstFrc, srcCinfo, rc2, interaction);
		}
		else
		{
			if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) continue;

			for (int cellX = max(cell0.x-1, 0); cellX <= min(cell0.x+1, srcCinfo.ncells.x-1); cellX++)
			{
				const int cid = srcCinfo.encode(cellX, cellY, cellZ);
				const int pstart = srcCinfo.cellStarts[cid];
				const int pend   = srcCinfo.cellStarts[cid+1];

				computeCell<NeedDstAcc, NeedSrcAcc, false> (pstart, pend, dstP, dstId, dstFrc, srcCinfo, rc2, interaction);
			}
		}

	if (NeedDstAcc)
		atomicAdd(dstView.forces + dstId, dstFrc);
}

/**
 * Compute interactions between particle of two different ParticleVector.
 *
 * Mapping is nine threads per particle. The rest is similar to
 * computeExternalInteractions_1tpp()
 */
template<bool NeedDstAcc, bool NeedSrcAcc, bool Variant, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions_9tpp(
		PVview dstView, CellListInfo srcCinfo,
		const float rc2, Interaction interaction)
{
	static_assert(NeedDstAcc || NeedSrcAcc, "External interactions should return at least some accelerations");

	const int gid = blockIdx.x*blockDim.x + threadIdx.x;

	const int dstId = gid / 9;
	const int dircode = gid % 9;

	if (dstId >= dstView.size) return;

	const Particle dstP(
			readNoCache(dstView.particles+2*dstId),
			readNoCache(dstView.particles+2*dstId+1) );

	float3 dstFrc = make_float3(0.0f);

	const int3 cell0 = srcCinfo.getCellIdAlongAxes<false>(dstP.r);

	int cellZ = cell0.z + dircode / 3 - 1;
	int cellY = cell0.y + dircode % 3 - 1;

	if (Variant)
	{
		if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) return;

		const int midCellId = srcCinfo.encode(cell0.x, cellY, cellZ);
		int rowStart  = max(midCellId-1, 0);
		int rowEnd    = min(midCellId+2, srcCinfo.totcells);

		const int pstart = srcCinfo.cellStarts[rowStart];
		const int pend   = srcCinfo.cellStarts[rowEnd];

		computeCell<NeedDstAcc, NeedSrcAcc, false> (pstart, pend, dstP, dstId, dstFrc, srcCinfo, rc2, interaction);
	}
	else
	{
		if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) return;

		for (int cellX = max(cell0.x-1, 0); cellX <= min(cell0.x+1, srcCinfo.ncells.x-1); cellX++)
		{
			const int cid = srcCinfo.encode(cellX, cellY, cellZ);
			const int pstart = srcCinfo.cellStarts[cid];
			const int pend   = srcCinfo.cellStarts[cid+1];

			computeCell<NeedDstAcc, NeedSrcAcc, false> (pstart, pend, dstP, dstId, dstFrc, srcCinfo, rc2, interaction);
		}
	}

	if (NeedDstAcc)
		atomicAdd(dstView.forces + dstId, dstFrc);
}

/**
 * Compute interactions between particle of two different ParticleVector.
 *
 * Mapping is 27 threads per particle. The rest is similar to
 * computeExternalInteractions_1tpp()
 */
template<bool NeedDstAcc, bool NeedSrcAcc, bool Variant, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions_27tpp(
		PVview dstView, CellListInfo srcCinfo,
		const float rc2, Interaction interaction)
{
	static_assert(NeedDstAcc || NeedSrcAcc, "External interactions should return at least some accelerations");

	const int gid = blockIdx.x*blockDim.x + threadIdx.x;

	const int dstId = gid / 27;
	const int dircode = gid % 27;

	if (dstId >= dstView.size) return;

	const Particle dstP(
			readNoCache(dstView.particles+2*dstId),
			readNoCache(dstView.particles+2*dstId+1) );

	float3 dstFrc = make_float3(0.0f);

	const int3 cell0 = srcCinfo.getCellIdAlongAxes<false>(dstP.r);

	int cellZ = cell0.z +  dircode / 9      - 1;
	int cellY = cell0.y + (dircode / 3) % 3 - 1;
	int cellX = cell0.x +  dircode % 3      - 1;

	if ( !( cellX >= 0 && cellX < srcCinfo.ncells.x &&
			cellY >= 0 && cellY < srcCinfo.ncells.y &&
			cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) return;

	const int cid = srcCinfo.encode(cellX, cellY, cellZ);
	const int pstart = srcCinfo.cellStarts[cid];
	const int pend   = srcCinfo.cellStarts[cid+1];

	computeCell<NeedDstAcc, NeedSrcAcc, false> (pstart, pend, dstP, dstId, dstFrc, srcCinfo, rc2, interaction);

	if (NeedDstAcc)
		atomicAdd(dstView.forces + dstId, dstFrc);
}
