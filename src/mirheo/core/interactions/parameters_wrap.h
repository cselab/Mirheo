#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/logger.h>

#include <extern/variant/include/mpark/variant.hpp>

#include <map>
#include <string>
#include <vector>

class ParametersWrap
{
public:

    using VarParam = mpark::variant<real, std::vector<real>, std::vector<real2>, std::string, bool>;
    using MapParams = std::map<std::string, VarParam>;
    
    ParametersWrap(const MapParams& params) :
        params(params)
    {
        for (const auto& p : params)
            readParams[p.first] = false;
    }

    template <typename T>
    bool exists(const std::string& key)
    {
        auto it = params.find(key);

        if (it == params.end())
            return false;

        if (!mpark::holds_alternative<T>(it->second))
            return false;

        return true;
    }

    void checkAllRead() const
    {
        for (const auto& p : readParams)
            if (p.second == false)
                die("invalid parameter '%s'", p.first.c_str());
    }

    template <typename T>
    T read(const std::string& key)
    {
        return read(key, Identity<T>());
    }

private:

    // have to do template specialization trick because explicit
    // specializations have to be at namespace scope inc C++
    
    template<typename T>
    struct Identity { using Type = T; };
    
    template <typename T>
    T read(const std::string& key, Identity<T>)
    {
        auto it = params.find(key);
    
        if (it == params.end())
            die("missing parameter '%s'", key.c_str());

        if (!mpark::holds_alternative<T>(it->second))
            die("'%s': invalid type", key.c_str());

        readParams[key] = true;
        return mpark::get<T>(it->second);
    }

    real2 read(const std::string& key, Identity<real2>)
    {
        const auto v = read<std::vector<real>>(key);
        if (v.size() != 2)
            die("%s must have 2 components", key.c_str());
        return {v[0], v[1]};
    }

    real3 read(const std::string& key, Identity<real3>)
    {
        const auto v = read<std::vector<real>>(key);
        if (v.size() != 3)
            die("%s must have 3 components", key.c_str());
        return {v[0], v[1], v[2]};
    }

private:
    const MapParams& params;
    std::map<std::string, bool> readParams;
};
