#include "interface.h"

namespace mirheo
{

Wall::Wall(const MirState *state, const std::string& name) :
    MirSimulationObject(state, name)
{}

Wall::~Wall() = default;

void Wall::setPrerequisites(__UNUSED ParticleVector *pv)
{}

SDF_basedWall::~SDF_basedWall() = default;

} // namespace mirheo
