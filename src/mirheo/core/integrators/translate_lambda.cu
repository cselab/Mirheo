#include "translate_lambda.h"
#include "integration_kernel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

IntegratorTranslateLambda::IntegratorTranslateLambda(const MirState *state,
                                                     const std::string& name,
                                                     std::function<real3(real)> vel) :
    Integrator(state, name),
    vel_(std::move(vel))
{}

IntegratorTranslateLambda::~IntegratorTranslateLambda() = default;

void IntegratorTranslateLambda::execute(ParticleVector *pv, cudaStream_t stream)
{
    const auto t = static_cast<real>(getState()->currentTime);

    const real3 vel = vel_(t);

    auto func = [vel] __device__ (Particle& p, real3 f, real invm, real dt)
    {
        p.u = vel;
        p.r += p.u * dt;
    };

    integrate(pv, getState()->getDt(), func, stream);
    invalidatePV_(pv);
}

} // namespace mirheo
