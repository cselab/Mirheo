#include "interface.h"

Wall::Wall(const YmrState *state, std::string name) :
    YmrSimulationObject(state, name)
{}

Wall::~Wall() = default;

void Wall::setPrerequisites(ParticleVector *pv)
{}

SDF_basedWall::~SDF_basedWall() = default;
