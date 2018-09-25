#include "channel.h"

#include <core/logger.h>

namespace XDMF
{
    Channel::Channel(std::string name, void* data, Type type, int entrySize_bytes, std::string typeStr) :
        name(name), data((float*)data), type(type),
        entrySize_floats(entrySize_bytes / sizeof(float)), typeStr(typeStr)
    {
        if (entrySize_floats*sizeof(float) != entrySize_bytes)
            die("Channel('%s') should have a chunk size in bytes divisible by %d (got %d)",
                name.c_str(), sizeof(float), entrySize_bytes);
    }

    std::string type_to_string(Channel::Type type)
    {
        switch (type) {
        case Channel::Type::Scalar:  return "Scalar";
        case Channel::Type::Vector:  return "Vector";
        case Channel::Type::Tensor6: return "Tensor6";
        case Channel::Type::Tensor9: return "Tensor9";
        case Channel::Type::Other:   return "Scalar";
        }
    }

    Channel::Type string_to_type(std::string str)
    {
        if (str == "Scalar")  return Channel::Type::Scalar;
        if (str == "Vector")  return Channel::Type::Vector;
        if (str == "Tensor6") return Channel::Type::Tensor6;
        if (str == "Tensor6") return Channel::Type::Tensor9;            
        return Channel::Type::Other;
    }

    int get_ncomponents(Channel::Type type)
    {
        switch (type) {
        case Channel::Type::Scalar:  return 1;
        case Channel::Type::Vector:  return 3;
        case Channel::Type::Tensor6: return 6;
        case Channel::Type::Tensor9: return 9;
        case Channel::Type::Other:   return 1;
        }
    }
}
