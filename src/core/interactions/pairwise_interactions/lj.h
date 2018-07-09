#pragma once

#include <core/datatypes.h>

class ParticleVector;
class CellList;


class Pairwise_LJ
{
public:
    Pairwise_LJ(float rc, float sigma, float epsilon, float maxForce) :
        rc(rc), sigma(sigma), epsilon(epsilon), maxForce(maxForce)
    {
        epsx24_sigma = 24.0*epsilon/sigma;
        rc2 = rc*rc;
    }

    void setup(LocalParticleVector* pv1, LocalParticleVector* pv2, CellList* cl1, CellList* cl2, float t)
    {    }

    __device__ inline float3 operator()(Particle dst, int dstId, Particle src, int srcId) const
    {
        const float3 dr = dst.r - src.r;
        const float rij2 = dot(dr, dr);

        if (rij2 > rc2) return make_float3(0.0f);

        const float rs2 = sigma*sigma / rij2;
        const float rs4 = rs2*rs2;
        const float rs8 = rs4*rs4;
        const float rs14 = rs8*rs4*rs2;

        const float IfI = epsx24_sigma * (2*rs14 - rs8);

        return dr * min(max(IfI, 0.0f), maxForce);
    }

private:

    float maxForce;
    float sigma, epsilon, rc;
    float epsx24_sigma, rc2;
};
