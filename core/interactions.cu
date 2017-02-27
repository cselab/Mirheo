#include <core/dpd-rng.h>
#include <core/containers.h>
#include <core/interaction_engine.h>
#include <core/helper_math.h>
#include <core/interactions.h>

//==================================================================================================================
// DPD interactions
//==================================================================================================================

template<int s>
inline __device__ float viscosity_function(float x)
{
    return sqrtf(viscosity_function<s - 1>(x));
}

template<> inline __device__ float viscosity_function<1>(float x) { return sqrtf(max(x, 1e-20f)); }
template<> inline __device__ float viscosity_function<0>(float x) { return x; }

__device__ __forceinline__ float3 dpd_interaction(
		const float3 dstCoo, const float3 dstVel, const int dstId,
		const float3 srcCoo, const float3 srcVel, const int srcId,
		const float adpd, const float gammadpd, const float sigmadpd,
		const float rc2, const float invrc, const float seed)
{
	const float3 dr = dstCoo - srcCoo;
	const float rij2 = dot(dr, dr);
	if (rij2 > rc2) return make_float3(0.0f);

	const float invrij = rsqrtf(max(rij2, 1e-20f));
	const float rij = rij2 * invrij;
	const float argwr = 1.0f - rij*invrc;
	const float wr = viscosity_function<0>(argwr);

	const float3 dr_r = dr * invrij;
	const float rdotv = dot(dr_r, (dstVel - srcVel));

	const float myrandnr = Logistic::mean0var1(seed, min(srcId, dstId), max(srcId, dstId));

	const float strength = adpd * argwr - (gammadpd * wr * rdotv + sigmadpd * myrandnr) * wr;

	return dr_r * strength;
}


void interactionDPD (InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream,
		float adpd, float gammadpd, float sigma_dt, float rc)
{
	auto dpdCore = [=] __device__ ( const float4 dstCoo, const float4 dstVel, const int dstId,
									const float4 srcCoo, const float4 srcVel, const int srcId)
	{
		return dpd_interaction( make_float3(dstCoo), make_float3(dstVel), dstId,
								make_float3(srcCoo), make_float3(srcVel), srcId,
								adpd, gammadpd, sigma_dt, rc*rc, 1.0/rc, t);
	};

	const int nth = 32 * 4;

	if (type == InteractionType::Regular)
	{
		// Self interaction
		if (pv1 == pv2)
		{
			if (pv1->np > 0)
			{
				debug2("Computing internal forces for %s (%d particles)", pv1->name.c_str(), pv1->np);
				computeSelfInteractions<<< (pv1->np + nth - 1) / nth, nth, 0, stream >>>(
						(float4*)pv1->coosvels.devPtr(), (float*)pv1->forces.devPtr(), cl->cellInfo(), cl->cellsStart.devPtr(), pv1->np, dpdCore);
			}
		}
		else // External interaction
		{
			if (pv1->np > 0 && pv2->np > 0)
			{
				debug2("Computing external forces for %s - %s (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), pv1->np, pv2->np);
				computeExternalInteractions<true, true> <<< (pv2->np + nth - 1) / nth, nth, 0, stream >>>(
											(float4*)pv2->coosvels.devPtr(), nullptr, (float4*)pv1->coosvels.devPtr(),
											(float*)pv1->forces.devPtr(), cl->cellInfo(), cl->cellsStart.devPtr(), pv2->np, dpdCore);
			}
		}
	}

	// Halo interaction
	if (type == InteractionType::Halo)
	{
		if (pv1->np > 0 && pv2->np > 0)
		{
			debug2("Computing halo forces for %s - %s(halo) (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), pv1->np, pv2->halo.size());
			computeExternalInteractions<false, true> <<< (pv2->halo.size() + nth - 1) / nth, nth, 0, stream >>>(
										(float4*)pv2->halo.devPtr(), nullptr, (float4*)pv1->coosvels.devPtr(),
										(float*)pv1->forces.devPtr(), cl->cellInfo(), cl->cellsStart.devPtr(), pv2->halo.size(), dpdCore);
		}
	}

}
