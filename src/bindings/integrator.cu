#include "integrators.h"

#include <core/integrators/vv.h>
#include <core/integrators/forcing_terms/none.h>

PyVelocityVerlet::PyVelocityVerlet(std::string name, float dt, std::tuple<float, float, float> f) :
    PyIntegrator(new IntegratorVV<Forcing_None>(name, dt, Forcing_None()))
{    }
