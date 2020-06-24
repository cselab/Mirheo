// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory.h"
#include <mirheo/core/mesh/membrane.h>
#include <mirheo/core/mesh/mesh.h>
#include <mirheo/core/utils/config.h>

namespace mirheo
{

std::shared_ptr<Mesh> loadMesh(const MirState *, Loader& loader,
                               const ConfigObject& config)
{
    const std::string& type = config["__type"];
    if (type == "Mesh")
        return std::make_shared<Mesh>(loader, config);
    if (type == "MembraneMesh")
        return std::make_shared<MembraneMesh>(loader, config);
    die("Unrecognized or unimplemented mesh type \"%s\".", type.c_str());
}

} // namespace mirheo
