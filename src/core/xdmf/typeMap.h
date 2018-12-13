#pragma once

#include <core/utils/typeMap.h>
#include "channel.h"

namespace XDMF
{
    template <typename T> Channel::DataForm getType() {return Channel::DataForm::Other;}

    template <> Channel::DataForm getType<float> () {return Channel::DataForm::Scalar;}
    template <> Channel::DataForm getType<double>() {return Channel::DataForm::Scalar;}
    template <> Channel::DataForm getType<int>   () {return Channel::DataForm::Scalar;}
    template <> Channel::DataForm getType<float3>() {return Channel::DataForm::Vector;}


    template <typename T> Channel::Datatype getDatatype() {return Channel::Datatype::Float;}

    template <> Channel::Datatype getDatatype<double>() {return Channel::Datatype::Double;}
    template <> Channel::Datatype getDatatype<int>   () {return Channel::Datatype::Int;}
}
