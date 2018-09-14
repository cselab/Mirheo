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
}
