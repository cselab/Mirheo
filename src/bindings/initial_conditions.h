#pragma once

#include <string>

#include "pycontainer.h"

class InitialConditions;
using PyInitialConditions = PyContainer<InitialConditions>;


class PyUniformIC : public PyInitialConditions
{
public:
	PyUniformIC(float density);
};


