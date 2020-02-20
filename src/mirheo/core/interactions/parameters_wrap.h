#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/logger.h>

#include <extern/variant/include/mpark/variant.hpp>

#include <map>
#include <string>
#include <vector>

namespace mirheo
{

/** \brief A tool to transform a map from string keys to variant parameters.
    
    The input map is typically an input from the python interface.
 */
class ParametersWrap
{
public:
    /// A variant that contains the possible types to represent parameters
    using VarParam = mpark::variant<real, std::vector<real>, std::vector<real2>, std::string, bool>;
    /// Represents the map from parameter names to parameter values
    using MapParams = std::map<std::string, VarParam>;

    /// \brief Construct a ParametersWrap object from a MapParams
    ParametersWrap(const MapParams& params);

    /** \brief Check if a parameter of a given type and name exists in the map
        \tparam T The type of the parameter
        \param [in] key The name of the parameter to check
        \return true if T and key match, false otherwise. 
     */
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

    /// \brief Die if some keys were not read (see read())
    void checkAllRead() const;

    /** \brief Fetch a parameter value for a given key.
        \tparam T the type of the parameter to read.
        \param [in] key the parameter name to read.

        On success, this method will also mark internally the parameter as read.
        This allows to check if some parameters were never used (see checkAllRead()).

        This method dies if key does not exist or if T is the wrong type.
    */
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
