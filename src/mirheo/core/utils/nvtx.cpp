#include "nvtx.h"

#ifdef USE_NVTX
#include <functional>

namespace mirheo
{

namespace NvtxHelpers
{
constexpr uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
constexpr int num_colors = sizeof(colors) / sizeof(colors[0]);
} // namespace NvtxHelpers

NvtxTracer::NvtxTracer(const std::string& name)
{
    std::hash<std::string> nameHash;
    
    int color_id = nameHash(name);
    color_id = color_id % NvtxHelpers::num_colors;
    
    nvtxEventAttributes_t event = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    event.version       = NVTX_VERSION;
    event.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.colorType     = NVTX_COLOR_ARGB;
    event.color         = NvtxHelpers::colors[color_id];
    event.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = getCName();

    id = nvtxRangeStartEx(&event);
}

NvtxTracer::~NvtxTracer()
{
    nvtxRangeEnd(id);
}

} // namespace mirheo

#endif
