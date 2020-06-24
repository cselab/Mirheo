#pragma once

#include <mirheo/core/types/variant_type_wrapper.h>

#include <hdf5.h>
#include <string>

namespace mirheo
{

namespace XDMF
{
/** \brief Describes one array of data to be dumped or read.
 */
struct Channel
{
    /// The "topology" of one element
    enum class DataForm { Scalar, Vector, Tensor6, Tensor9, Quaternion, Triangle, Vector4, RigidMotion, Other };
    /// The type of the data contained in one element
    enum class NumberType { Float, Double, Int, Int64 };
    /// If the data depends on the coordinates
    enum class NeedShift { True, False };

    std::string name;       ///< Name of the channel
    void *data;             ///< pointer to the data that needs to be dumped
    DataForm dataForm;      ///< topology of one element
    NumberType numberType;  ///< data type (enum version)
    TypeDescriptor type;    ///< data type (variant version)
    NeedShift needShift;    ///< wether the data depends on the coordinates or not

    int nComponents() const; ///< Number of component in each element (e.g. Vector has 3)
    int precision() const;   ///< Number of bytes of each component in one element
};

/// \return the xdmf-compatible string that describes the Channel::DataForm
std::string dataFormToXDMFAttribute (Channel::DataForm dataForm);
/// \return the number of components in the Channel::DataForm
int         dataFormToNcomponents   (Channel::DataForm dataForm);
/// \return a unique string that describes the Channel::DataForm (two different may map to the same xdmf attribute)
std::string dataFormToDescription   (Channel::DataForm dataForm);

/// reverse of dataFormToDescription()
Channel::DataForm descriptionToDataForm(const std::string& str);

/// \return the HDF5-compatible description of the given Channel::NumberType data type
decltype (H5T_NATIVE_FLOAT) numberTypeToHDF5type  (Channel::NumberType dt);
/// \return the xdmf-compatible string corresponding to the given Channel::NumberType data type
std::string                 numberTypeToString    (Channel::NumberType dt);
/// \return the size in bytes of the type represented by the given Channel::NumberType data type
int                         numberTypeToPrecision (Channel::NumberType dt);

/// reverse of numberTypeToString() and numberTypeToPrecision()
Channel::NumberType infoToNumberType(const std::string& str, int precision);

} // namespace XDMF

} // namespace mirheo
