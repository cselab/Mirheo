#include "const_omega.h"
#include "integration_kernel.h"

#include <core/utils/kernel_launch.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>


IntegratorConstOmega::IntegratorConstOmega(std::string name, float dt, pyfloat3 center, pyfloat3 omega) :
    Integrator(name, dt),
    center(make_float3(center)), omega(make_float3(omega))
{   }

IntegratorConstOmega::~IntegratorConstOmega() = default;

void IntegratorConstOmega::stage1(ParticleVector* pv, float t, cudaStream_t stream)
{   }

void IntegratorConstOmega::stage2(ParticleVector* pv, float t, cudaStream_t stream)
{
    const auto domain = pv->domain;
    const auto _center = center;
    const auto _omega = omega;

    auto rotate = [domain, _center, _omega] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
        float3 gr = domain.local2global(p.r);
        p.u = cross(_omega, gr - _center);
        float IrI = length(p.r);
        p.r += p.u*dt;

        p.r = normalize(p.r) * IrI;
    };

    int nthreads = 128;

    // New particles now become old
    std::swap(pv->local()->coosvels, *pv->local()->extraPerParticle.getData<Particle>("old_particles"));
    PVviewWithOldParticles pvView(pv, pv->local());

    SAFE_KERNEL_LAUNCH(
            integrationKernel,
            getNblocks(2*pvView.size, nthreads), nthreads, 0, stream,
            pvView, dt, rotate );

    // PV may have changed, invalidate all
    pv->haloValid = false;
    pv->redistValid = false;
    pv->cellListStamp++;
}
