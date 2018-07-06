#include "vv_periodicPoiseuille.h"
#include "forcing_terms/periodic_poiseuille.h"
#include "vv.h"
#include <core/utils/make_unique.h>


void IntegratorVV_periodicPoiseuille::stage1(ParticleVector* pv, float t, cudaStream_t stream)
{
    impl-> stage1(pv, t, stream);
}
void IntegratorVV_periodicPoiseuille::stage2(ParticleVector* pv, float t, cudaStream_t stream)
{
    impl-> stage2(pv, t, stream);
}

IntegratorVV_periodicPoiseuille::IntegratorVV_periodicPoiseuille(std::string name, float dt, float force, std::string direction) :
    Integrator(name, dt)
{
    Forcing_PeriodicPoiseuille::Direction dir;
    if      (direction == "x") dir = Forcing_PeriodicPoiseuille::Direction::x;
    else if (direction == "y") dir = Forcing_PeriodicPoiseuille::Direction::y;
    else if (direction == "z") dir = Forcing_PeriodicPoiseuille::Direction::z;
    else
        die("Direction passed to periodic poiseuille integrator '%s' should be x, y or z, but got:",
            name.c_str(), direction.c_str());
        
    Forcing_PeriodicPoiseuille term(force, dir);
    impl = std::make_unique<IntegratorVV<Forcing_PeriodicPoiseuille>> (name, dt, term);
}

IntegratorVV_periodicPoiseuille::~IntegratorVV_periodicPoiseuille() = default;
