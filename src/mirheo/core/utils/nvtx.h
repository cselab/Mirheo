#pragma once

#ifdef USE_NVTX
#include <string>
#include <nvToolsExt.h>

namespace mirheo
{

class NvtxTracer
{
public:
    NvtxTracer(const std::string& name);
    ~NvtxTracer();
private:
    nvtxRangeId_t id;
};

} // namespace mirheo

#define NvtxCreateRange(identifyer, name) NvtxTracer identifyer(name)
#else
#define NvtxCreateRange(identifyer, name)
#endif
