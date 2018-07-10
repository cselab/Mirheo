#include "vv_noforce.h"
#include "forcing_terms/none.h"
#include "vv.h"
#include <core/utils/make_unique.h>


void IntegratorVV_noforce::stage1(ParticleVector* pv, float t, cudaStream_t stream)
{
    impl-> stage1(pv, t, stream);
}
void IntegratorVV_noforce::stage2(ParticleVector* pv, float t, cudaStream_t stream)
{
    impl-> stage2(pv, t, stream);
}

IntegratorVV_noforce::IntegratorVV_noforce(std::string name, float dt, std::tuple<float, float, float> extra_force) :
    Integrator(name, dt)
{
    Forcing_None term;
    impl = std::make_unique<IntegratorVV<Forcing_None>> (name, dt, term);
}

IntegratorVV_noforce::~IntegratorVV_noforce() = default;
