#include <core/dpd-rng.h>
#include <core/particle_vector.h>
#include <core/interaction_engine.h>
#include <core/helper_math.h>
#include <core/interactions.h>
#include <core/cuda_common.h>
#include <core/object_vector.h>

//==================================================================================================================
// Interaction wrapper macro
//==================================================================================================================

#define WRAP_INTERACTON(INTERACTION_FUNCTION) \
	if (type == InteractionType::Regular)                                                                                                          \
	{                                                                                                                                              \
		/*  Self interaction */                                                                                                                    \
		if (pv1 == pv2)                                                                                                                            \
		{                                                                                                                                          \
			debug2("Computing internal forces for %s (%d particles)", pv1->name.c_str(), pv1->local()->size());                                    \
                                                                                                                                                   \
			const int nth = 128;                                                                                                                   \
			if (pv1->local()->size() > 0)                                                                                                          \
				computeSelfInteractions<<< (pv1->local()->size() + nth - 1) / nth, nth, 0, stream >>>(                                             \
						pv1->local()->size(), (float4*)cl->coosvels->devPtr(), (float*)cl->forces->devPtr(),                                       \
						cl->cellInfo(), cl->cellsStartSize.devPtr(), rc*rc, INTERACTION_FUNCTION);                                                 \
		}                                                                                                                                          \
		else /*  External interaction */                                                                                                           \
		{                                                                                                                                          \
			debug2("Computing external forces for %s - %s (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), pv1->local()->size(), pv2->local()->size());           \
                                                                                                                                                   \
			const int nth = 128;                                                                                                                   \
			if (pv1->local()->size() > 0 && pv2->local()->size() > 0)                                                                                                        \
				computeExternalInteractions<true, true, true> <<< (pv2->local()->size() + nth - 1) / nth, nth, 0, stream >>>(                                   \
						pv2->local()->size(),                                                                                                                   \
						(float4*)pv2->local()->coosvels.devPtr(), (float*)pv2->local()->forces.devPtr(),                                                             \
						(float4*)cl->coosvels->devPtr(), (float*)cl->forces->devPtr(),                                                             \
						cl->cellInfo(), cl->cellsStartSize.devPtr(),                                                                               \
						rc*rc, INTERACTION_FUNCTION);                                                                                              \
		}                                                                                                                                          \
	}                                                                                                                                              \
                                                                                                                                                   \
	/*  Halo interaction */                                                                                                                        \
	if (type == InteractionType::Halo)                                                                                                             \
	{                                                                                                                                              \
		debug2("Computing halo forces for %s - %s(halo) (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), pv1->local()->size(), pv2->halo()->size());    \
                                                                                                                                                   \
		const int nth = 128;                                                                                                                       \
		if (pv1->local()->size() > 0 && pv2->halo()->size() > 0)                                                                                   \
			computeExternalInteractions<false, true, false> <<< (pv2->halo()->size() + nth - 1) / nth, nth, 0, stream >>>(                         \
					pv2->halo()->size(),                                                                                                           \
					(float4*)pv2->halo()->coosvels.devPtr(), nullptr,                                                                              \
					(float4*)cl->coosvels->devPtr(), (float*)cl->forces->devPtr(),                                                                 \
					cl->cellInfo(), cl->cellsStartSize.devPtr(),                                                                                   \
					rc*rc, INTERACTION_FUNCTION);                                                                                                  \
	}



//==================================================================================================================
// DPD interactions
//==================================================================================================================

inline __device__ float viscosityKernel(const float x, const float k)
{
	if (k == 1.0f)   return x;
	if (k == 0.5f)   return sqrtf(max(x, 1e-10f));
	if (k == 0.25f)  return sqrtf(sqrtf(max(x, 1e-10f)));
	if (k == 0.125f) return sqrtf(sqrtf(sqrtf(max(x, 1e-10f))));

    return powf(x, k);
}

__device__ __forceinline__ float3 pairwiseDPD(
		Particle dst, Particle src,
		const float adpd, const float gammadpd, const float sigmadpd,
		const float rc2, const float invrc, const float k, const float seed)
{
	const float3 dr = dst.r - src.r;
	const float rij2 = dot(dr, dr);
	if (rij2 > rc2) return make_float3(0.0f);

	const float invrij = rsqrtf(max(rij2, 1e-20f));
	const float rij = rij2 * invrij;
	const float argwr = 1.0f - rij*invrc;
	const float wr = viscosityKernel(argwr, k);

	const float3 dr_r = dr * invrij;
	const float rdotv = dot(dr_r, (dst.u - src.u));

	const float myrandnr = Logistic::mean0var1(seed, min(src.i1, dst.i1), max(src.i1, dst.i1));

	const float strength = adpd * argwr - (gammadpd * wr * rdotv + sigmadpd * myrandnr) * wr;

	return dr_r * strength;
}


void interactionDPD (InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream,
		float adpd, float gammadpd, float sigma_dt, float power, float rc)
{
	// Better to use random number in the seed instead of periodically changing time
	const float seed = drand48();
	auto dpdCore = [=] __device__ ( Particle dst, Particle src )
	{
		return pairwiseDPD( dst, src, adpd, gammadpd, sigma_dt, rc*rc, 1.0/rc, power, seed);
	};

	WRAP_INTERACTON(dpdCore)
}


//==================================================================================================================
// LJ interactions
//==================================================================================================================

__device__ inline float3 pairwiseLJ(Particle dst, Particle src, const float sigma, const float epsx24_sigma, const float rc2)
{
	const float3 dr = dst.r - src.r;
	const float rij2 = dot(dr, dr);

	if (rij2 > rc2) return make_float3(0.0f);

	const float rs2 = sigma*sigma / rij2;
	const float rs4 = rs2*rs2;
	const float rs8 = rs4*rs4;
	const float rs14 = rs8*rs4*rs2;

	return dr * epsx24_sigma * (2*rs14 - rs8);
}

__device__ inline float3 pairwiseLJ_objectAware(Particle dst, Particle src,
		bool isDstObj, float3 dstCom,
		bool isSrcObj, float3 srcCom,
		const float sigma, const float epsx24_sigma, const float rc2)
{
	const float3 dr = dst.r - src.r;

	const bool dstSide = dot(dr, dst.r-dstCom) < 0.0f;
	const bool srcSide = dot(dr, srcCom-src.r) < 0.0f;

	if (dstSide && (!isSrcObj)) return make_float3(0.0f);
	if ((!isDstObj) && srcSide) return make_float3(0.0f);
	if (dstSide && srcSide)     return make_float3(0.0f);

	return pairwiseLJ(dst, src, sigma, epsx24_sigma, rc2);
}


void interactionLJ_objectAware(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream,
		float epsilon, float sigma, float rc)
{
	auto ov1 = dynamic_cast<ObjectVector*>(pv1);
	auto ov2 = dynamic_cast<ObjectVector*>(pv2);
	if (ov1 == nullptr && ov2 == nullptr)
		die("Object-aware LJ interaction can only be used with objects");

	const float epsx24_sigma = 24.0*epsilon/sigma;
	const float rc2 = rc*rc;
	const bool self = pv1 == pv2;

	const LocalObjectVector::COMandExtent* dstComExt = (ov1 != nullptr) ? ov1->local()->comAndExtents.devPtr() : nullptr;
	const LocalObjectVector::COMandExtent* srcComExt = (ov2 != nullptr) ? ov2->local()->comAndExtents.devPtr() : nullptr;

	auto dpdCore = [=] __device__ ( Particle dst, Particle src )
	{
		const int dstObjId = dst.s21;
		const int srcObjId = src.s21;

		if (dstObjId == srcObjId && self) return make_float3(0.0f);

		float3 dstCom = make_float3(0.0f);
		float3 srcCom = make_float3(0.0f);
		if (dstComExt != nullptr) dstCom = dstComExt[dstObjId].com;
		if (srcComExt != nullptr) srcCom = srcComExt[srcObjId].com;

		return pairwiseLJ_objectAware( dst, src, (dstComExt != nullptr), dstCom, (srcComExt != nullptr), srcCom, sigma, epsx24_sigma, rc2);
	};

	WRAP_INTERACTON(dpdCore)
}



















