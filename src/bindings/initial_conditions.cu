#include "initial_conditions.h"

#include <core/initial_conditions/uniform_ic.h>

PyUniformIC::PyUniformIC(float density) :
    PyInitialConditions(new UniformIC(density))
{    }

