#include "translate.h"
#include "integration_kernel.h"

#include <core/utils/kernel_launch.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>


/**
 * @param vel Move with this velocity
 */
IntegratorTranslate::IntegratorTranslate(std::string name, float dt, pyfloat3 vel) :
    Integrator(name, dt), vel(make_float3(vel))
{    }

void IntegratorTranslate::stage2(ParticleVector* pv, float t, cudaStream_t stream)
{
    const auto _vel = vel;

    auto translate = [_vel] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
        p.u = _vel;
        p.r += p.u*dt;
    };

    int nthreads = 128;

    // New particles now become old
    std::swap(pv->local()->coosvels, *pv->local()->extraPerParticle.getData<Particle>("old_particles"));
    PVviewWithOldParticles pvView(pv, pv->local());

    SAFE_KERNEL_LAUNCH(
            integrationKernel,
            getNblocks(2*pvView.size, nthreads), nthreads, 0, stream,
            pvView, dt, translate );

    // PV may have changed, invalidate all
    pv->haloValid = false;
    pv->redistValid = false;
    pv->cellListStamp++;
}
