#pragma once

#include "type_list.h"

#include <extern/variant/include/mpark/variant.hpp>

#include <string>

namespace mirheo
{

template<class T>
struct DataTypeWrapper {using type = T;};

using TypeDescriptor = mpark::variant<
#define MAKE_WRAPPER(a) DataTypeWrapper<a>
    MIRHEO_TYPE_TABLE_COMMA(MAKE_WRAPPER)
#undef MAKE_WRAPPER
    >;

std::string typeDescriptorToString(const TypeDescriptor& desc);
TypeDescriptor stringToTypeDescriptor(const std::string& str);

} // namespace mirheo
