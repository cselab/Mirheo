#pragma once

#include <mirheo/core/types/variant_type_wrapper.h>

#include <hdf5.h>
#include <string>

namespace mirheo
{

namespace XDMF
{
struct Channel
{
    enum class DataForm { Scalar, Vector, Tensor6, Tensor9, Quaternion, Triangle, Vector4, RigidMotion, Other };
    enum class NumberType { Float, Double, Int, Int64 };
    enum class NeedShift { True, False };
    
    std::string name;
    void *data;
    DataForm dataForm;
    NumberType numberType;
    TypeDescriptor type;
    NeedShift needShift;

    Channel(std::string name, void *data, DataForm dataForm,
            NumberType numberType, TypeDescriptor type, NeedShift needShift);

    int nComponents() const;
    int precision() const;
};
    
std::string dataFormToXDMFAttribute (Channel::DataForm dataForm);
int         dataFormToNcomponents   (Channel::DataForm dataForm);
std::string dataFormToDescription   (Channel::DataForm dataForm);

Channel::DataForm descriptionToDataForm(const std::string& str);
    

decltype (H5T_NATIVE_FLOAT) numberTypeToHDF5type  (Channel::NumberType dt);
std::string                 numberTypeToString    (Channel::NumberType dt);
int                         numberTypeToPrecision (Channel::NumberType dt);

Channel::NumberType infoToNumberType(const std::string& str, int precision);

} // namespace XDMF

} // namespace mirheo
