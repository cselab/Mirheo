#pragma once

#include "channel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/types/type_list.h>

namespace mirheo
{

namespace XDMF
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS

/// \return The Channel::DataForm from the template parameter
template <typename T> Channel::DataForm inline getDataForm()
{
    error("DataForm not implemented");
    return Channel::DataForm::Other;
}


#define IMPLEMENT_DATAFORM(type)                                        \
    template <> Channel::DataForm inline getDataForm<type>      () {return Channel::DataForm::Scalar;} \
    template <> Channel::DataForm inline getDataForm<type ## 3> () {return Channel::DataForm::Vector;} \
    template <> Channel::DataForm inline getDataForm<type ## 4> () {return Channel::DataForm::Vector4;}

IMPLEMENT_DATAFORM(int)
IMPLEMENT_DATAFORM(float)
IMPLEMENT_DATAFORM(double)

template <> Channel::DataForm inline getDataForm<int64_t> () {return Channel::DataForm::Scalar;}
template <> Channel::DataForm inline getDataForm<RigidMotion> () {return Channel::DataForm::RigidMotion;}

#undef IMPLEMENT_DATAFORM


/// \return The Channel::NumberType from the template parameter
template <typename T> Channel::NumberType getNumberType();

template <> Channel::NumberType inline getNumberType<float>  () {return Channel::NumberType::Float;}
template <> Channel::NumberType inline getNumberType<double> () {return Channel::NumberType::Double;}
template <> Channel::NumberType inline getNumberType<int>    () {return Channel::NumberType::Int;}
template <> Channel::NumberType inline getNumberType<int64_t>() {return Channel::NumberType::Int64;}
template <> Channel::NumberType inline getNumberType<Stress> () {return getNumberType<decltype(Stress::xx)>();}

template <typename T>
Channel::NumberType inline getNumberType()
{
    using BaseType = decltype(T::x);
    return getNumberType<BaseType>();
}

template <> Channel::NumberType inline getNumberType<TemplRigidMotion<double>> () {return getNumberType<double>();}
template <> Channel::NumberType inline getNumberType<TemplRigidMotion<float>>  () {return getNumberType<float>();}
template <> Channel::NumberType inline getNumberType<COMandExtent>             () {return getNumberType<decltype(COMandExtent::com)>();}
template <> Channel::NumberType inline getNumberType<Force>                    () {return getNumberType<decltype(Force::f)>();}

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace XDMF

} // namespace mirheo
