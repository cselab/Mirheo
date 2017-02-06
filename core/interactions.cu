#include "dpd-rng.h"
#include "containers.h"
#include "interaction_engine.h"
#include "helper_math.h"

//==================================================================================================================
// DPD interactions
//==================================================================================================================

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
		const float adpd, const float gammadpd, const float sigmadpd,
		const float rc2, const float invrc, const float seed)
{
	const float _xr = dstCoo.x - srcCoo.x;
	const float _yr = dstCoo.y - srcCoo.y;
	const float _zr = dstCoo.z - srcCoo.z;
	const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	if (rij2 > rc2) return make_float3(0.0f);

	const float invrij = rsqrtf(rij2);
	const float rij = rij2 * invrij;
	const float argwr = 1.0f - rij*invrc;
	const float wr = viscosity_function<0>(argwr);

	const float xr = _xr * invrij;
	const float yr = _yr * invrij;
	const float zr = _zr * invrij;

	const float rdotv =
			xr * (dstVel.x - srcVel.x) +
			yr * (dstVel.y - srcVel.y) +
			zr * (dstVel.z - srcVel.z);

	const float myrandnr = Logistic::mean0var1(seed, min(srcId, dstId), max(srcId, dstId));

	const float strength = adpd * argwr - (gammadpd * wr * rdotv + sigmadpd * myrandnr) * wr;

	return make_float3(strength * xr, strength * yr, strength * zr);
}


void interactionDPDSelf (ParticleVector* pv, CellList* cl, const float t, cudaStream_t stream,
		float adpd, float gammadpd, float sigma_dt, float rc)
{
	auto dpdCore = [=] __device__ ( const float3 dstCoo, const float3 dstVel, const int dstId,
					   const float3 srcCoo, const float3 srcVel, const int srcId)
	{
		return dpd_interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId, adpd, gammadpd, sigma_dt, rc*rc, 1.0f/rc, t);
	};

	const int nth = 32 * 4;

	if (pv->np > 0)
	{
		debug("Computing internal forces for %s (%d particles)", pv->name.c_str(), pv->np);
		computeSelfInteractions<<< (pv->np + nth - 1) / nth, nth, 0, stream >>>(
				(float4*)pv->coosvels.devPtr(), (float*)pv->forces.devPtr(), cl->cellInfo(), cl->cellsStart.devPtr(), pv->np, dpdCore);
	}
}

void interactionDPDHalo (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream,
		float adpd, float gammadpd, float sigma_dt, float rc)
{
	auto dpdCore = [=] __device__ ( const float3 dstCoo, const float3 dstVel, const int dstId,
					   const float3 srcCoo, const float3 srcVel, const int srcId)
	{
		return dpd_interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId, adpd, gammadpd, sigma_dt, rc*rc, 1.0f/rc, t);
	};

	const int nth = 32 * 4;

	if (pv1->np > 0 && pv2->np > 0)
	{
		debug("Computing halo forces for %s - %s(halo) (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), pv1->np, pv2->halo.size());
		computeExternalInteractions<false, true> <<< (pv2->halo.size() + nth - 1) / nth, nth, 0, stream >>>(
									(float4*)pv2->halo.devPtr(), nullptr, (float4*)pv1->coosvels.devPtr(),
									(float*)pv1->forces.devPtr(), cl->cellInfo(), cl->cellsStart.devPtr(), pv2->halo.size(), dpdCore);
	}
}

void interactionDPDExternal (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream,
		float adpd, float gammadpd, float sigma_dt, float rc)
{
	auto dpdCore = [=] __device__ ( const float3 dstCoo, const float3 dstVel, const int dstId,
					   const float3 srcCoo, const float3 srcVel, const int srcId)
	{
		return dpd_interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId, adpd, gammadpd, sigma_dt, rc*rc, 1.0f/rc, t);
	};

	const int nth = 32 * 4;

	if (pv1->np > 0 && pv2->np > 0)
	{
		debug("Computing external forces for %s - %s (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), pv1->np, pv2->np);
		computeExternalInteractions<true, true> <<< (pv2->np + nth - 1) / nth, nth, 0, stream >>>(
									(float4*)pv2->coosvels.devPtr(), nullptr, (float4*)pv1->coosvels.devPtr(),
									(float*)pv1->forces.devPtr(), cl->cellInfo(), cl->cellsStart.devPtr(), pv2->np, dpdCore);
	}
}
