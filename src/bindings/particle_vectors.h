#pragma once

#include <string>

#include "pycontainer.h"

class ParticleVector;
using PyParticleVector = PyContainer<ParticleVector>;


class PySimpleParticleVector : public PyParticleVector
{
public:
	PySimpleParticleVector(std::string name, float mass);
};

