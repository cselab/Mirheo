#include "parameters_wrap.h"

namespace mirheo
{

ParametersWrap::ParametersWrap(const ParametersWrap::MapParams& params) :
    params_(params)
{
    for (const auto& p : params_)
        readParams_[p.first] = false;
}

void ParametersWrap::checkAllRead() const
{
    for (const auto& p : readParams_)
        if (p.second == false)
            die("invalid parameter '%s'", p.first.c_str());
}

real2 ParametersWrap::_read(const std::string& key, ParametersWrap::Identity<real2>)
{
    const auto v = read<std::vector<real>>(key);
    if (v.size() != 2)
        die("%s must have 2 components", key.c_str());
    return {v[0], v[1]};
}

real3 ParametersWrap::_read(const std::string& key, ParametersWrap::Identity<real3>)
{
    const auto v = read<std::vector<real>>(key);
    if (v.size() != 3)
        die("%s must have 3 components", key.c_str());
    return {v[0], v[1], v[2]};
}

} // namespace mirheo
