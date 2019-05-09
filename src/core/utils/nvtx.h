#pragma once

#ifdef USE_NVTX
#include "nvToolsExt.h"

class NvtxTracer
{
public:
    NvtxTracer(const char *name);
    ~NvtxTracer();
private:
    nvtxRangeId_t id;
};

#define NvtxCreateRange(identifyer, name) NvtxTracer identifyer(name)
#else
#define NvtxCreateRange(identifyer, name)
#endif
