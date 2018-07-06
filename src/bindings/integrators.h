#pragma once

#include <string>
#include "pycontainer.h"

class Integrator;
using PyIntegrator = PyContainer<Integrator>;

class PyVelocityVerlet : public PyIntegrator
{
public:
	PyVelocityVerlet(std::string name, float dt, std::tuple<float, float, float> ff);
};

