#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/logger.h>

#include <extern/variant/include/mpark/variant.hpp>

#include <map>
#include <string>
#include <vector>

namespace mirheo
{

class ParametersWrap
{
public:

    using VarParam = mpark::variant<real, std::vector<real>, std::vector<real2>, std::string, bool>;
    using MapParams = std::map<std::string, VarParam>;
    
    ParametersWrap(const MapParams& params);

    template <typename T>
    bool exists(const std::string& key)
    {
        auto it = params_.find(key);

        if (it == params_.end())
            return false;

        if (!mpark::holds_alternative<T>(it->second))
            return false;

        return true;
    }

    void checkAllRead() const;

    template <typename T>
    T read(const std::string& key)
    {
        return _read(key, Identity<T>());
    }

private:

    // have to do template specialization trick because explicit
    // specializations have to be at namespace scope in C++
    
    template<typename T>
    struct Identity { using Type = T; };
    
    template <typename T>
    T _read(const std::string& key, Identity<T>)
    {
        auto it = params_.find(key);
    
        if (it == params_.end())
            die("missing parameter '%s'", key.c_str());

        if (!mpark::holds_alternative<T>(it->second))
            die("'%s': invalid type", key.c_str());

        readParams_[key] = true;
        return mpark::get<T>(it->second);
    }

    real2 _read(const std::string& key, Identity<real2>);
    real3 _read(const std::string& key, Identity<real3>);

private:
    const MapParams& params_;
    std::map<std::string, bool> readParams_;
};

} // namespace mirheo
