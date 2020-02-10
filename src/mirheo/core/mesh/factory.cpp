#include "factory.h"
#include <mirheo/core/mesh/membrane.h>
#include <mirheo/core/mesh/mesh.h>

namespace mirheo
{

std::shared_ptr<Mesh> loadMesh(const MirState *, Loader& loader,
                               const ConfigObject& config, const std::string &type)
{
    if (type == "Mesh")
        return std::make_shared<Mesh>(loader, config);
    if (type == "MembraneMesh")
        return std::make_shared<MembraneMesh>(loader, config);
    die("Unrecognized or unimplemented mesh type \"%s\".", type.c_str());
}

} // namespace mirheo
