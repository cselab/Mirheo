#pragma once

#include <string>

namespace XDMF
{
    struct Channel
    {
        std::string name;
        std::string typeStr;
        float* data;
        int entrySize_floats;
        
        enum class Type
        {
            Scalar, Vector, Tensor6, Tensor9, Other
        } type;
        
        Channel(std::string name, void* data, Type type, int entrySize_bytes, std::string typeStr = "float");
    };

    Channel::Type string_to_type(std::string str);
    std::string type_to_string(Channel::Type type);
    int get_ncomponents(Channel::Type type);
}
