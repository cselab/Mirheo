#pragma once

#include "channel.h"

#include <core/utils/type_map.h>

namespace XDMF
{
template <typename T> Channel::DataForm inline getDataForm() {return Channel::DataForm::Other;}

template <> Channel::DataForm inline getDataForm<float> () {return Channel::DataForm::Scalar;}
template <> Channel::DataForm inline getDataForm<double>() {return Channel::DataForm::Scalar;}
template <> Channel::DataForm inline getDataForm<int>   () {return Channel::DataForm::Scalar;}
template <> Channel::DataForm inline getDataForm<float3>() {return Channel::DataForm::Vector;}


template <typename T> Channel::NumberType inline getNumberType() {return Channel::NumberType::Float;}

template <> Channel::NumberType inline getNumberType<double>() {return Channel::NumberType::Double;}
template <> Channel::NumberType inline getNumberType<double3>() {return Channel::NumberType::Double;}
template <> Channel::NumberType inline getNumberType<double4>() {return Channel::NumberType::Double;}
template <> Channel::NumberType inline getNumberType<int>   () {return Channel::NumberType::Int;}
}
