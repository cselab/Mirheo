#include "vv_constDP.h"
#include "forcing_terms/const_dp.h"
#include "vv.h"
#include <core/utils/make_unique.h>


void IntegratorVV_constDP::stage1(ParticleVector* pv, float t, cudaStream_t stream)
{
    impl-> stage1(pv, t, stream);
}
void IntegratorVV_constDP::stage2(ParticleVector* pv, float t, cudaStream_t stream)
{
    impl-> stage2(pv, t, stream);
}

IntegratorVV_constDP::IntegratorVV_constDP(std::string name, float dt, pyfloat3 extra_force) :
    Integrator(name, dt)
{
    Forcing_ConstDP term(make_float3(extra_force));
    impl = std::make_unique<IntegratorVV<Forcing_ConstDP>> (name, dt, term);
}

IntegratorVV_constDP::~IntegratorVV_constDP() = default;
