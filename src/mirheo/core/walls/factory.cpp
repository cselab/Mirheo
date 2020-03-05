#include "factory.h"
#include "simple_stationary_wall.h"
#include <mirheo/core/utils/config.h>

namespace mirheo
{

static bool startsWith(const std::string &text, const char *tmp)
{
    return text.compare(0, strlen(tmp), tmp) == 0;
}

std::shared_ptr<Wall>
wall_factory::loadWall(const MirState *state, Loader& loader, const ConfigObject& config)
{
    const std::string& type = config["__type"];
    if (startsWith(type, "SimpleStationaryWall<"))
        return loadSimpleStationaryWall(state, loader, config);
    die("Unrecognized or unimplemented interaction type \"%s\".", type.c_str());
}

} // namespace mirheo
