#pragma once

#include <string>
#include <hdf5.h>

namespace XDMF
{
    struct Channel
    {
        std::string name;
        std::string typeStr;
        void* data;
        int entrySize_floats;
        
        enum class Type
        {
            Scalar, Vector, Tensor6, Tensor9, Other
        } type;
        
        enum class Datatype
        {
            Float, Int
        } datatype;
        
        Channel(std::string name, void* data, Type type, int entrySize_bytes,
                std::string typeStr = "float", Datatype datatype = Datatype::Float);
    };

    Channel::Type string_to_type(std::string str);
    std::string type_to_string(Channel::Type type);
    int get_ncomponents(Channel::Type type);
    
    decltype (H5T_NATIVE_FLOAT) datatypeToHDF5type(Channel::Datatype dt);
    std::string datatypeToString(Channel::Datatype dt);
    int datatypeToPrecision(Channel::Datatype dt);
    Channel::Datatype stringToDatatype(std::string str);
}
