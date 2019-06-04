#include "channel.h"

#include <core/logger.h>

namespace XDMF
{

Channel::Channel(std::string name, void *data, DataForm dataForm, NumberType numberType, TypeDescriptor type) :
    name(name),
    data(data),
    dataForm(dataForm),
    numberType(numberType),
    type(type)
{}

int Channel::nComponents() const
{
    return dataFormToNcomponents(dataForm);
}

int Channel::precision() const
{
    return numberTypeToPrecision(numberType);
}
    
std::string dataFormToXDMFAttribute(Channel::DataForm dataForm)
{
    switch (dataForm)
    {
    case Channel::DataForm::Scalar:     return "Scalar";
    case Channel::DataForm::Vector:     return "Vector";
    case Channel::DataForm::Tensor6:    return "Tensor6";
    case Channel::DataForm::Tensor9:    return "Tensor";
    case Channel::DataForm::Quaternion: return "Matrix";
    case Channel::DataForm::Triangle:   return "Matrix";
    case Channel::DataForm::Vector4:    return "Matrix";
    case Channel::DataForm::Other:      return "Scalar";
    }
}

int dataFormToNcomponents(Channel::DataForm dataForm)
{
    switch (dataForm)
    {
    case Channel::DataForm::Scalar:     return 1;
    case Channel::DataForm::Vector:     return 3;
    case Channel::DataForm::Tensor6:    return 6;
    case Channel::DataForm::Tensor9:    return 9;
    case Channel::DataForm::Quaternion: return 4;
    case Channel::DataForm::Triangle:   return 3;
    case Channel::DataForm::Vector4:    return 4;
    case Channel::DataForm::Other:      return 1;
    }
}

std::string dataFormToDescription(Channel::DataForm dataForm)
{
    switch (dataForm)
    {
    case Channel::DataForm::Scalar:     return "Scalar";
    case Channel::DataForm::Vector:     return "Vector";
    case Channel::DataForm::Tensor6:    return "Tensor6";
    case Channel::DataForm::Tensor9:    return "Tensor";
    case Channel::DataForm::Quaternion: return "Quaternion";
    case Channel::DataForm::Triangle:   return "Triangle";
    case Channel::DataForm::Vector4:    return "Vector4";
    case Channel::DataForm::Other:      return "Other";
    }
}
        
Channel::DataForm descriptionToDataForm(std::string str)
{
    if (str == "Scalar")      return Channel::DataForm::Scalar;
    if (str == "Vector")      return Channel::DataForm::Vector;
    if (str == "Tensor6")     return Channel::DataForm::Tensor6;
    if (str == "Tensor")      return Channel::DataForm::Tensor9;
    if (str == "Quaternion")  return Channel::DataForm::Quaternion;
    if (str == "Trianle")     return Channel::DataForm::Triangle;
    if (str == "Vector4")     return Channel::DataForm::Vector4;
    warn("Unrecognised format '%s'", str.c_str());
    return Channel::DataForm::Other;
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
}
    
std::string numberTypeToString(Channel::NumberType dt)
{
    switch (dt)
    {
    case Channel::NumberType::Float  : return "Float";
    case Channel::NumberType::Double : return "Float";
    case Channel::NumberType::Int    : return "Int";
    case Channel::NumberType::Int64  : return "Int";
    }
}

int numberTypeToPrecision(Channel::NumberType dt)
{
    switch (dt)
    {
    case Channel::NumberType::Float  : return sizeof(float);
    case Channel::NumberType::Double : return sizeof(double);
    case Channel::NumberType::Int    : return sizeof(int);
    case Channel::NumberType::Int64  : return sizeof(int64_t);
    }
}
    
Channel::NumberType infoToNumberType(std::string str, int precision)
{
    if (precision == sizeof(float)   && str == "Float") return Channel::NumberType::Float;
    if (precision == sizeof(double)  && str == "Float") return Channel::NumberType::Double;
    if (precision == sizeof(int)     && str == "Int")   return Channel::NumberType::Int;
    if (precision == sizeof(int64_t) && str == "Int")   return Channel::NumberType::Int64;
    die("NumberType '%s' with precision %d is not supported for reading", str.c_str(), precision);
}

} // namespace XDMF
