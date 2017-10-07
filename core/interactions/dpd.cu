#include "dpd.h"

#include <core/cuda_common.h>
#include <core/celllist.h>
#include <core/cuda-rng.h>

#include "pairwise_engine.h"
#include "wrapper_macro.h"

__device__ __forceinline__ float fastPower(const float x, const float k)
{
	if (fabs(k - 1.0f)   < 1e-6f) return x;
	if (fabs(k - 0.5f)   < 1e-6f) return sqrtf(fabs(x));
	if (fabs(k - 0.25f)  < 1e-6f) return sqrtf(fabs(sqrtf(fabs(x))));
	if (fabs(k - 0.125f) < 1e-6f) return sqrtf(fabs(sqrtf(fabs(sqrtf(fabs(x))))));

    return powf(fabs(x), k);
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
	const float wr = fastPower(argwr, k);

	const float3 dr_r = dr * invrij;
	const float3 du = dst.u - src.u;
	const float rdotv = dot(dr_r, du);

	const float myrandnr = 0*Logistic::mean0var1(seed, min(src.i1, dst.i1), max(src.i1, dst.i1));

	const float strength = adpd * argwr - (gammadpd * wr * rdotv + sigmadpd * myrandnr) * wr;

	return dr_r * strength;
}


InteractionDPD::InteractionDPD(std::string name, float rc, float a, float gamma, float kbT, float dt, float power) :
		Interaction(name, rc), a(a), gamma(gamma), power(power)
{
	sigma = sqrt(2 * gamma * kbT / dt);
}

void InteractionDPD::_compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
{
	// Better to use random number in the seed instead of periodically changing time
	const float seed = drand48();
	const float rc2 = rc*rc;
	const float rc_1 = 1.0 / rc;
	const float _a = a;
	const float _gamma = gamma;
	const float _sigma = sigma;
	const float _power = power;
	auto dpdCore = [=] __device__ ( Particle dst, Particle src ) {
		return pairwiseDPD( dst, src, _a, _gamma, _sigma, rc2, rc_1, _power, seed);
	};

	WRAP_INTERACTON(dpdCore)
}
