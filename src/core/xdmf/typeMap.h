#pragma once

#include <core/utils/typeMap.h>
#include "channel.h"

namespace XDMF
{
    template <typename T> Channel::Type getType() {return Channel::Type::Other;}

    template <> Channel::Type getType<float> () {return Channel::Type::Scalar;}
    template <> Channel::Type getType<double>() {return Channel::Type::Scalar;}
    template <> Channel::Type getType<int>   () {return Channel::Type::Scalar;}
    template <> Channel::Type getType<float3>() {return Channel::Type::Vector;}


    template <typename T> Channel::Datatype getDatatype() {return Channel::Datatype::Float;}

    template <> Channel::Datatype getDatatype<double>() {return Channel::Datatype::Double;}
    template <> Channel::Datatype getDatatype<int>   () {return Channel::Datatype::Int;}
}
