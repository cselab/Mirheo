#pragma once

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_rng.h>

#include <random>

class ParticleVector;
class CellList;


__device__ inline float fastPower(const float x, const float k)
{
	if (fabsf(k - 1.0f)   < 1e-6f) return x;
	if (fabsf(k - 0.5f)   < 1e-6f) return sqrtf(fabsf(x));
	if (fabsf(k - 0.25f)  < 1e-6f) return sqrtf(sqrtf(fabsf(x)));
	if (fabsf(k - 0.125f) < 1e-6f) return sqrtf(sqrtf(sqrtf(fabsf(x))));

    return powf(fabsf(x), k);
}


class Pairwise_DPD
{
public:
	Pairwise_DPD(float rc, float a, float gamma, float kbT, float dt, float power) :
		rc(rc), a(a), gamma(gamma), power(power)
	{
		sigma = sqrt(2 * gamma * kbT / dt);
		rc2 = rc*rc;
		invrc = 1.0 / rc;
	}

	void setup(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, float t)
	{
		//seed = t;
		// better use random seed (time-based!) instead of time
		// t is float, use it's bit representation as int to seed RNG
		int v = *((int*)&t);
		std::mt19937 gen(v);
		std::uniform_real_distribution<float> udistr(0.001, 1);
		seed = udistr(gen);
	}

	__device__ inline float3 operator()(const Particle dst, int dstId, const Particle src, int srcId) const
	{
		const float3 dr = dst.r - src.r;
		const float rij2 = dot(dr, dr);
		if (rij2 > 1.0f) return make_float3(0.0f);

		const float invrij = rsqrtf(rij2);
		const float rij = rij2 * invrij;
		const float argwr = 1.0f - rij;
		const float wr = fastPower(argwr, power);

		const float3 dr_r = dr * invrij;
		const float3 du = dst.u - src.u;
		const float rdotv = dot(dr_r, du);

		const float myrandnr = Logistic::mean0var1(seed, min(src.i1, dst.i1), max(src.i1, dst.i1));

		const float strength = a * argwr - (gamma * wr * rdotv + sigma* myrandnr) * wr;

		return dr_r * strength;
	}

private:

	float a, gamma, sigma, power, rc;
	float invrc, rc2;
	float seed;
};
