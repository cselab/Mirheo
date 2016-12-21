#pragma once

#include "datatypes.h"
#include "containers.h"

#include <string>
#include <unordered_map>

using PVHash = std::unordered_map<std::string, ParticleVector>;

struct InteractionDesc
{
	CellList* cellList;
	std::function exec;
};

class Simulation
{
private:

	std::unordered_map<std::string, int> PVname2index;

	std::vector<ParticleVector*> particleVectors;
	std::vector<std::vector<InteractionDesc>> interactionTable;
};
