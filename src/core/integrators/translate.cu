#include "translate.h"
#include "integration_kernel.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>


/**
 * @param vel Move with this velocity
 */
IntegratorTranslate::IntegratorTranslate(const MirState *state, std::string name, real3 vel) :
    Integrator(state, name),
    vel(vel)
{}

IntegratorTranslate::~IntegratorTranslate() = default;

void IntegratorTranslate::stage2(ParticleVector *pv, cudaStream_t stream)
{
    const auto _vel = vel;

    auto translate = [_vel] __device__ (Particle& p, const real3 f, const real invm, const real dt) {
        p.u = _vel;
        p.r += p.u*dt;
    };

    integrate(pv, state->dt, translate, stream);
    invalidatePV(pv);
}
