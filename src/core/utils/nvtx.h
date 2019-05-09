#pragma once

#ifdef USE_NVTX
#include <string>
#include <nvToolsExt.h>

class NvtxTracer
{
public:
    NvtxTracer(const std::string& name);
    ~NvtxTracer();
private:
    nvtxRangeId_t id;
};

#define NvtxCreateRange(identifyer, name) NvtxTracer identifyer(name)
#else
#define NvtxCreateRange(identifyer, name)
#endif
