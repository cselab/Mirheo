// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#ifdef USE_NVTX
#include <string>
#include <nvToolsExt.h>

namespace mirheo
{

/** \brief a RAII class that allows to profile a scope with NVTX ranges

    The NVTX range starts at the construction of this object.
    It ends when the destructor of this object is called.
 */
class NvtxTracer
{
public:
    /** \brief Start an NVTX range
        \param name The name of the range as it will appear in the profiling information

        Internally, a color is set to the range from the hash of \p name.
     */
    NvtxTracer(const std::string& name);

    /// end the NVTX range
    ~NvtxTracer();
private:
    nvtxRangeId_t id_;
};

} // namespace mirheo

#define NvtxCreateRange(identifyer, name) NvtxTracer identifyer(name)
#else
#define NvtxCreateRange(identifyer, name)
#endif
