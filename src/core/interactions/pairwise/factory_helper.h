#pragma once

#include "kernels/parameters.h"

#include <core/interactions/parameters_wrap.h>

#include <limits>

namespace FactoryHelper
{
constexpr auto defaultFloat = std::numeric_limits<float>::infinity();

struct ParamsReader
{
    enum Mode {FailIfNotFound, DefaultIfNotFound};
    
    template <typename T>
    T read(ParametersWrap& desc, const std::string& key) const
    {
        if (mode == Mode::FailIfNotFound)
        {
            return desc.read<T>(key);
        }
        else
        {
            if (desc.exists<T>(key))
                return desc.read<T>(key);
            else
                return makeDefault<T>();
        }
    }

    Mode mode {Mode::FailIfNotFound};

    template <class T> T makeDefault() const;
};


template <class Params> void readParams(Params& p, ParametersWrap& desc, ParamsReader reader);

DPDParams       readDPDParams     (ParametersWrap& desc);
LJParams        readLJParams      (ParametersWrap& desc);
MDPDParams      readMDPDParams    (ParametersWrap& desc);
DensityParams   readDensityParams (ParametersWrap& desc);
SDPDParams      readSDPDParams    (ParametersWrap& desc);

VarStressParams readStressParams  (ParametersWrap& desc);

} // FactoryHelper
