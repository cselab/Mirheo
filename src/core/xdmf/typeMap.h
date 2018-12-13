#pragma once

#include <core/utils/typeMap.h>
#include "channel.h"

namespace XDMF
{
    template <typename T> Channel::DataForm getDataForm() {return Channel::DataForm::Other;}

    template <> Channel::DataForm getDataForm<float> () {return Channel::DataForm::Scalar;}
    template <> Channel::DataForm getDataForm<double>() {return Channel::DataForm::Scalar;}
    template <> Channel::DataForm getDataForm<int>   () {return Channel::DataForm::Scalar;}
    template <> Channel::DataForm getDataForm<float3>() {return Channel::DataForm::Vector;}


    template <typename T> Channel::NumberType getNumberType() {return Channel::NumberType::Float;}

    template <> Channel::NumberType getNumberType<double>() {return Channel::NumberType::Double;}
    template <> Channel::NumberType getNumberType<int>   () {return Channel::NumberType::Int;}
}
