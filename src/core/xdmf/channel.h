#pragma once

#include <core/utils/type_map.h>

#include <hdf5.h>
#include <string>

namespace XDMF
{
struct Channel
{
    enum class DataForm { Scalar, Vector, Tensor6, Tensor9, Quaternion, Triangle, Other };
    enum class NumberType { Float, Double, Int, Int64 };
    
    std::string name;
    void *data;
    DataForm dataForm;
    NumberType numberType;
    TypeDescriptor type;

    Channel(std::string name, void *data, DataForm dataForm, NumberType numberType, TypeDescriptor type);
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

} // namespace XDMF
