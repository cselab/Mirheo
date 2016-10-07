#include "dpd.h"
#include "dpd-rng.h"
#include "interaction_engine.h"
#include "logger.h"

template<int s>
inline __device__ float viscosity_function(float x)
{
    return sqrtf(viscosity_function<s - 1>(x));
}

template<> inline __device__ float viscosity_function<1>(float x) { return sqrtf(x); }
template<> inline __device__ float viscosity_function<0>(float x){ return x; }

__device__ __forceinline__ float3 dpd_interaction(
		const float3 dstCoo, const float3 dstVel, const int dstId,
		const float3 srcCoo, const float3 srcVel, const int srcId,
		float adpd, float gammadpd, float sigmadpd, float seed)
{
	const float _xr = dstCoo.x - srcCoo.x;
	const float _yr = dstCoo.y - srcCoo.y;
	const float _zr = dstCoo.z - srcCoo.z;
	const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	if (rij2 > 1.0f) return make_float3(0.0f);

	const float invrij = rsqrtf(rij2);
	const float rij = rij2 * invrij;
	const float argwr = 1.0f - rij;
	const float wr = viscosity_function<0>(argwr);

	const float xr = _xr * invrij;
	const float yr = _yr * invrij;
	const float zr = _zr * invrij;

	const float rdotv =
			xr * (dstVel.x - srcVel.x) +
			yr * (dstVel.y - srcVel.y) +
			zr * (dstVel.z - srcVel.z);

	const float myrandnr = 0*Logistic::mean0var1(seed, min(srcId, dstId), max(srcId, dstId));

	const float strength = adpd * argwr - (gammadpd * wr * rdotv + sigmadpd * myrandnr) * wr;

	return make_float3(strength * xr, strength * yr, strength * zr);
}

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

	CUDA_Check( cudaMemsetAsync(pv.accs.devdata, 0, sizeof(float4)* pv.np, stream) );
	const int nth = 128;

	debug("Computing internal forces for %d paricles", pv.np);
	computeSelfInteractions<<< (pv.np + nth - 1) / nth, nth, 0, stream >>>(
			(float4*)pv.coosvels.devdata, (float*)pv.accs.devdata, pv.cellsStart.devdata, pv.cellsSize.devdata,
			pv.ncells, pv.domainStart, pv.totcells+1, pv.np, dpdInt);
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

	const int nth = 128;
	debug("Computing halo forces for %d ext paricles", pv.halo.size);
	computeHaloInteractions<false, true> <<< (pv.halo.size + nth - 1) / nth, nth, 0, stream >>>(
			(float4*)pv.halo.devdata, nullptr, (float4*)pv.coosvels.devdata, (float*)pv.accs.devdata, pv.cellsStart.devdata,
				pv.ncells, pv.domainStart, pv.totcells+1, pv.halo.size, dpdInt);
}
