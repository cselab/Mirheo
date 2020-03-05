#pragma once

#include "kernels/parameters.h"

#include <mirheo/core/interactions/utils/parameters_wrap.h>

#include <limits>

namespace mirheo
{

namespace factory_helper
{
constexpr auto defaultReal = std::numeric_limits<real>::infinity();

/// Helper to read parameters from a ParametersWrap object.
/// Two read modes can be used (see \c Mode)
struct ParamsReader
{
    /// The available read modes
    enum Mode
    {
        FailIfNotFound,   ///< Will fail if the required parameter is not found (default).
        DefaultIfNotFound ///< Will return a default parameter (infinit value) if required parameter is not found.
    };

    /** \brief Read a parameter from \p desc with key \p key
        \tparam The parameter type
        \param [in,out] desc The set of available parameters
        \param [in] key Name of the parameter to read
        \return The parameter value.

        This method will fail if the key and type do not match the parameters in \p desc in FailIfNotFound mode.
     */
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

    Mode mode {Mode::FailIfNotFound}; ///< read mode

    /// create a default value for a given type
    template <class T> T makeDefault() const;
};


template <class Params> void readParams(Params& p, ParametersWrap& desc, ParamsReader reader);

DPDParams       readDPDParams     (ParametersWrap& desc);
LJParams        readLJParams      (ParametersWrap& desc);
MDPDParams      readMDPDParams    (ParametersWrap& desc);
DensityParams   readDensityParams (ParametersWrap& desc);
SDPDParams      readSDPDParams    (ParametersWrap& desc);

VarStressParams readStressParams  (ParametersWrap& desc);



template <class Params>
void readSpecificParams(Params& p, ParametersWrap& desc)
{
    using namespace factory_helper;
    readParams(p, desc, {ParamsReader::Mode::DefaultIfNotFound});
}

void readSpecificParams(LJParams&      p, ParametersWrap& desc);
void readSpecificParams(DensityParams& p, ParametersWrap& desc);
void readSpecificParams(SDPDParams&    p, ParametersWrap& desc);

} // factory_helper

} // namespace mirheo
