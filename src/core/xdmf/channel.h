#pragma once

#include <string>
#include <hdf5.h>

namespace XDMF
{
    struct Channel
    {
        std::string name;
        void *data;
        
        enum class DataForm
        {
            Scalar, Vector, Tensor6, Tensor9, Quaternion, Triangle, Other
        } dataForm;
        
        enum class Datatype
        {
            Float, Int, Double
        } datatype;
        
        Channel(std::string name, void *data, DataForm dataForm, Datatype datatype = Datatype::Float);
        int nComponents() const;
        int precision() const;
    };
    
    std::string dataFormToXDMFAttribute (Channel::DataForm dataForm);
    int         dataFormToNcomponents   (Channel::DataForm dataForm);
    std::string dataFormToDescription   (Channel::DataForm dataForm);

    Channel::DataForm descriptionToDataForm(std::string str);
    

    decltype (H5T_NATIVE_FLOAT) datatypeToHDF5type  (Channel::Datatype dt);
    std::string                 datatypeToString    (Channel::Datatype dt);
    int                         datatypeToPrecision (Channel::Datatype dt);

    Channel::Datatype infoToDatatype(std::string str, int precision);
}
