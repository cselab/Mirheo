// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "channel.h"

#include <mirheo/core/logger.h>

namespace mirheo
{

namespace XDMF
{

int Channel::nComponents() const
{
    return dataFormToNcomponents(dataForm);
}

int Channel::precision() const
{
    return numberTypeToPrecision(numberType);
}

namespace details {
std::string dataFormToXDMFAttribute(Channel::Scalar)     { return "Scalar";}
std::string dataFormToXDMFAttribute(Channel::Vector)     { return "Vector";}
std::string dataFormToXDMFAttribute(Channel::Tensor6)    { return "Tensor6";}
std::string dataFormToXDMFAttribute(Channel::Tensor9)    { return "Tensor";}
std::string dataFormToXDMFAttribute(Channel::Quaternion) { return "Matrix";}
std::string dataFormToXDMFAttribute(Channel::Triangle)   { return "Matrix";}
std::string dataFormToXDMFAttribute(Channel::Vector4)    { return "Matrix";}
std::string dataFormToXDMFAttribute(Channel::RigidMotion){ return "Matrix";}
std::string dataFormToXDMFAttribute(Channel::Other)      { return "Scalar";}
} // namespace details

std::string dataFormToXDMFAttribute(Channel::DataForm dataForm)
{
    return std::visit([](auto &&val) {return details::dataFormToXDMFAttribute(val);}, dataForm);
}


namespace details {
int dataFormToNcomponents(Channel::Scalar)     { return 1;}
int dataFormToNcomponents(Channel::Vector)     { return 3;}
int dataFormToNcomponents(Channel::Tensor6)    { return 6;}
int dataFormToNcomponents(Channel::Tensor9)    { return 9;}
int dataFormToNcomponents(Channel::Quaternion) { return 4;}
int dataFormToNcomponents(Channel::Triangle)   { return 3;}
int dataFormToNcomponents(Channel::Vector4)    { return 4;}
int dataFormToNcomponents(Channel::RigidMotion)
{
    constexpr auto szRM = sizeof(RigidMotion);
    constexpr auto szRMx = sizeof(RigidMotion::r.x);
    static_assert(szRM % szRMx == 0, "RigidMotion components must be of same type");
    return szRM / szRMx ;
}
int dataFormToNcomponents(Channel::Other)      { return 1;}
} // namespace details

int dataFormToNcomponents(Channel::DataForm dataForm)
{
    return std::visit([](auto&& val){return details::dataFormToNcomponents(val);}, dataForm);
}

namespace details {
std::string dataFormToDescription(Channel::Scalar)     { return "Scalar";}
std::string dataFormToDescription(Channel::Vector)     { return "Vector";}
std::string dataFormToDescription(Channel::Tensor6)    { return "Tensor6";}
std::string dataFormToDescription(Channel::Tensor9)    { return "Tensor";}
std::string dataFormToDescription(Channel::Quaternion) { return "Quaternion";}
std::string dataFormToDescription(Channel::Triangle)   { return "Triangle";}
std::string dataFormToDescription(Channel::Vector4)    { return "Vector4";}
std::string dataFormToDescription(Channel::RigidMotion){ return "RigidMotion";}
std::string dataFormToDescription(Channel::Other)      { return "Other";}
} // namespace details

std::string dataFormToDescription(Channel::DataForm dataForm)
{
    return std::visit([](auto&& val){return details::dataFormToDescription(val);}, dataForm);
}

Channel::DataForm descriptionToDataForm(const std::string& str)
{
    if (str == "Scalar")      return Channel::Scalar{};
    if (str == "Vector")      return Channel::Vector{};
    if (str == "Tensor6")     return Channel::Tensor6{};
    if (str == "Tensor")      return Channel::Tensor9{};
    if (str == "Quaternion")  return Channel::Quaternion{};
    if (str == "Trianle")     return Channel::Triangle{};
    if (str == "Vector4")     return Channel::Vector4{};
    if (str == "RigidMotion") return Channel::RigidMotion{};
    warn("Unrecognised format '%s'", str.c_str());
    return Channel::Other{};
}

decltype (H5T_NATIVE_FLOAT) numberTypeToHDF5type(Channel::NumberType nt)
{
    switch (nt)
    {
    case Channel::NumberType::Float  : return H5T_NATIVE_FLOAT;
    case Channel::NumberType::Double : return H5T_NATIVE_DOUBLE;
    case Channel::NumberType::Int    : return H5T_NATIVE_INT;
    case Channel::NumberType::Int64  : return H5T_NATIVE_INT64;
    }
    return H5T_NATIVE_FLOAT;
}

std::string numberTypeToString(Channel::NumberType nt)
{
    switch (nt)
    {
    case Channel::NumberType::Float  : return "Float";
    case Channel::NumberType::Double : return "Float";
    case Channel::NumberType::Int    : return "Int";
    case Channel::NumberType::Int64  : return "Int";
    }
    return "Invalid";
}

int numberTypeToPrecision(Channel::NumberType nt)
{
    switch (nt)
    {
    case Channel::NumberType::Float  : return sizeof(float);
    case Channel::NumberType::Double : return sizeof(double);
    case Channel::NumberType::Int    : return sizeof(int);
    case Channel::NumberType::Int64  : return sizeof(int64_t);
    }
    return sizeof(float);
}

Channel::NumberType infoToNumberType(const std::string& str, int precision)
{
    if (precision == sizeof(float)   && str == "Float") return Channel::NumberType::Float;
    if (precision == sizeof(double)  && str == "Float") return Channel::NumberType::Double;
    if (precision == sizeof(int)     && str == "Int")   return Channel::NumberType::Int;
    if (precision == sizeof(int64_t) && str == "Int")   return Channel::NumberType::Int64;
    die("NumberType '%s' with precision %d is not supported for reading", str.c_str(), precision);
    return Channel::NumberType::Float;
}

} // namespace XDMF

} // namespace mirheo
