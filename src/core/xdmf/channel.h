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
        
        enum class NumberType
        {
            Float, Int, Double
        } numberType;
        
        Channel(std::string name, void *data, DataForm dataForm, NumberType numberType = NumberType::Float);
        int nComponents() const;
        int precision() const;
    };
    
    std::string dataFormToXDMFAttribute (Channel::DataForm dataForm);
    int         dataFormToNcomponents   (Channel::DataForm dataForm);
    std::string dataFormToDescription   (Channel::DataForm dataForm);

    Channel::DataForm descriptionToDataForm(std::string str);
    

    decltype (H5T_NATIVE_FLOAT) numberTypeToHDF5type  (Channel::NumberType dt);
    std::string                 numberTypeToString    (Channel::NumberType dt);
    int                         numberTypeToPrecision (Channel::NumberType dt);

    Channel::NumberType infoToNumberType(std::string str, int precision);
}
