#include "channel.h"

#include <core/logger.h>

namespace XDMF
{
    Channel::Channel(std::string name, void* data, Type type, int entrySize_bytes, Datatype datatype) :
        name(name), data((float*)data), type(type),
        entrySize_floats(entrySize_bytes / sizeof(float)),
        datatype(datatype)
    {
        if (entrySize_floats*sizeof(float) != entrySize_bytes)
            die("Channel('%s') should have a chunk size in bytes divisible by %d (got %d)",
                name.c_str(), sizeof(float), entrySize_bytes);
    }

    std::string typeToXDMFAttribute(Channel::Type type)
    {
        switch (type)
        {
            case Channel::Type::Scalar:     return "Scalar";
            case Channel::Type::Vector:     return "Vector";
            case Channel::Type::Tensor6:    return "Tensor6";
            case Channel::Type::Tensor9:    return "Tensor";
            case Channel::Type::Quaternion: return "Matrix";
            case Channel::Type::Other:      return "Scalar";
        }
    }

    int typeToNcomponents(Channel::Type type)
    {
        switch (type)
        {
            case Channel::Type::Scalar:     return 1;
            case Channel::Type::Vector:     return 3;
            case Channel::Type::Tensor6:    return 6;
            case Channel::Type::Tensor9:    return 9;
            case Channel::Type::Quaternion: return 4;
            case Channel::Type::Other:      return 1;
        }
    }

    std::string typeToDescription(Channel::Type type)
    {
        switch (type)
        {
            case Channel::Type::Scalar:     return "Scalar";
            case Channel::Type::Vector:     return "Vector";
            case Channel::Type::Tensor6:    return "Tensor6";
            case Channel::Type::Tensor9:    return "Tensor";
            case Channel::Type::Quaternion: return "Quaternion";
            case Channel::Type::Other:      return "Other";
        }
    }
        
    Channel::Type descriptionToType(std::string str)
    {
        if (str == "Scalar")      return Channel::Type::Scalar;
        if (str == "Vector")      return Channel::Type::Vector;
        if (str == "Tensor6")     return Channel::Type::Tensor6;
        if (str == "Tensor")      return Channel::Type::Tensor9;
        if (str == "Quaternion")  return Channel::Type::Quaternion;
        return Channel::Type::Other;
    }
        
    decltype (H5T_NATIVE_FLOAT) datatypeToHDF5type(Channel::Datatype dt)
    {
        switch (dt)
        {
            case Channel::Datatype::Float  : return H5T_NATIVE_FLOAT;
            case Channel::Datatype::Double : return H5T_NATIVE_DOUBLE;
            case Channel::Datatype::Int    : return H5T_NATIVE_INT;
        }
    }
    
    std::string datatypeToString(Channel::Datatype dt)
    {
        switch (dt)
        {
            case Channel::Datatype::Float  : return "Float";
            case Channel::Datatype::Double : return "Float";
            case Channel::Datatype::Int    : return "Int";
        }
    }

    int datatypeToPrecision(Channel::Datatype dt)
    {
        switch (dt)
        {
            case Channel::Datatype::Float  : return sizeof(float);
            case Channel::Datatype::Double : return sizeof(double);
            case Channel::Datatype::Int    : return sizeof(int);
        }
    }
    
    Channel::Datatype infoToDatatype(std::string str, int precision)
    {
        if (precision == sizeof(float)  && str == "Float") return Channel::Datatype::Float;
        if (precision == sizeof(double) && str == "Float") return Channel::Datatype::Double;
        if (precision == sizeof(int)    && str == "Int")   return Channel::Datatype::Int;
        die("Datatype '%s' with precision %d is not supported for reading", str.c_str(), precision);
    }
}
