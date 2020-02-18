#include "interface.h"

namespace mirheo
{

Wall::Wall(const MirState *state, const std::string& name) :
    MirSimulationObject(state, name)
{}

Wall::~Wall() = default;

void Wall::setPrerequisites(__UNUSED ParticleVector *pv)
{}

SDFBasedWall::~SDFBasedWall() = default;

} // namespace mirheo
