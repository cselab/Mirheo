#pragma once

#include <memory>
#include <string>

namespace mirheo
{

class ConfigObject;
class MirState;
class Mesh;
class Loader;

/// Load mesh snapshot.
std::shared_ptr<Mesh> loadMesh(const MirState *state, Loader& loader,
                               const ConfigObject& config, const std::string &type);

} // namespace mirheo
