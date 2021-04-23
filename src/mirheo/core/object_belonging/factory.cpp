// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory.h"
#include "mesh_belonging.h"
#include <mirheo/core/utils/config.h>

namespace mirheo
{

namespace object_belonging_checker_factory
{

std::shared_ptr<ObjectBelongingChecker> loadChecker(
        const MirState *state, Loader&, const ConfigObject& config)
{
    // To keep the classes clean, we parse the constructor arguments here.
    const std::string& type = config["__type"];
    const std::string& name = config["name"];
    if (type == "MeshBelongingChecker")
        return std::make_shared<MeshBelongingChecker>(state, name);
    die("Unrecognized or unimplemented object belonging checker type \"%s\".", type.c_str());
}

} // namespace object_belonging_checker_factory
} // namespace mirheo
