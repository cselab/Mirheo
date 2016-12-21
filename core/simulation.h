#pragma once

#include "datatypes.h"
#include "containers.h"

#include <string>
#include <unordered_map>

using PVHash = std::unordered_map<std::string, ParticleVector>;


class Simulation
{
private:

	std::unordered_map<std::string, int> PVname2index;
	std::vector<ParticleVector*> particleVectors;
};
