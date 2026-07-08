// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "type_list.h"

#include <string>
#include <variant>

namespace mirheo
{
/** \brief A simple structure to store a c type.
    \tparam T The type to wrap

    This is useful with a variant and visitor pattern.
 */
template<class T>
struct DataTypeWrapper
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS // breathe warnings
    using type = T; ///< The wrapped type
#endif // DOXYGEN_SHOULD_SKIP_THIS
};

/// a variant of the type list available in data channels.
/// See also DataTypeWrapper.
using TypeDescriptor = std::variant<
#define MAKE_WRAPPER(a) DataTypeWrapper<a>
    MIRHEO_TYPE_TABLE_COMMA(MAKE_WRAPPER)
#undef MAKE_WRAPPER
    >;

/** \brief Convert a TypeDescriptor variant to the string that represents the type.
    \param desc The variant of DataTypeWrapper
    \return The string that correspond to the type (e.g. int gives "int")
*/
std::string typeDescriptorToString(const TypeDescriptor& desc);

/** \brief reverse operation of typeDescriptorToString().
    \param str The string representation of the type (e.g. "int" for int)
    \return a variant that contains the DataTypeWrapper with the correct type.

    This method will die if \p str does not correspond to any type in the type list.
 */
TypeDescriptor stringToTypeDescriptor(const std::string& str);

} // namespace mirheo
