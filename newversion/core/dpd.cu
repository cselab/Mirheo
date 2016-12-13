#include "dpd.h"
#include "dpd-rng.h"
#include "interaction_engine.h"
#include "interactions.h"
#include "logger.h"

void computeInternalDPD(ParticleVector& pv, cudaStream_t stream)
{
	const float dt = 0.0025;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigmadpd = sqrt(2 * gammadpd * kBT);
	const float adpd = 50;
	const float seed = 1.0f;

	const float sigma_dt = sigmadpd / sqrt(dt);
	auto dpdInt = [=] __device__ ( const float3 dstCoo, const float3 dstVel, const int dstId,
					   const float3 srcCoo, const float3 srcVel, const int srcId) {
		return dpd_interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId,
			adpd, gammadpd, sigma_dt, seed);
	};

	cudaFuncSetCacheConfig( computeSelfInteractions<decltype(dpdInt)>, cudaFuncCachePreferL1 );

	const int nth = 32 * 4;

	if (pv.np > 0)
	{
		debug("Computing internal forces for %d paricles", pv.np);
		computeSelfInteractions<<< (pv.np + nth - 1) / nth, nth, 0, stream >>>(
				(float4*)pv.coosvels.devdata, (float*)pv.accs.devdata, pv.cellsStart.devdata, pv.cellsSize.devdata,
				pv.ncells, pv.domainStart, pv.ncells.x*pv.ncells.y*pv.ncells.z+1, pv.np, dpdInt);
	}
}

void computeHaloDPD(ParticleVector& pv, cudaStream_t stream)
{
	const float dt = 0.0025;
	const float kBT = 1.0;
	const float gammadpd = 20;
	const float sigmadpd = sqrt(2 * gammadpd * kBT);
	const float adpd = 50;
	const float seed = 1.0f;

	const float sigma_dt = sigmadpd / sqrt(dt);
	auto dpdInt = [=] __device__ ( const float3 dstCoo, const float3 dstVel, const int dstId,
					   const float3 srcCoo, const float3 srcVel, const int srcId) {
		return dpd_interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId,
			adpd, gammadpd, sigma_dt, seed);
	};

	if (pv.halo.size > 0)
	{
		const int nth = 128;
		debug("Computing halo forces for %d ext paricles", pv.halo.size);
		computeHaloInteractions<false, true> <<< (pv.halo.size + nth - 1) / nth, nth, 0, stream >>>(
					(float4*)pv.halo.devdata, nullptr, (float4*)pv.coosvels.devdata, (float*)pv.accs.devdata, pv.cellsStart.devdata,
					pv.ncells, pv.domainStart, pv.totcells+1, pv.halo.size, dpdInt);
	}
}
