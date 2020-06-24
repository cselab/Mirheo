#include "variant_type_wrapper.h"

#include <mirheo/core/logger.h>

namespace mirheo
{

struct VisitorToStr
{
#define TYPE2STR(Type) std::string operator()(const DataTypeWrapper<Type>&) const {return #Type ;}

    MIRHEO_TYPE_TABLE(TYPE2STR)

#undef TYPE2STR
};

std::string typeDescriptorToString(const TypeDescriptor& desc)
{
    return mpark::visit(VisitorToStr(), desc);
}

TypeDescriptor stringToTypeDescriptor(const std::string& str)
{
#define IF_ENTRY(Type) if (str == #Type) return { DataTypeWrapper<Type>() };

    MIRHEO_TYPE_TABLE(IF_ENTRY);

#undef IF_ENTRY

    die("Unrecognized type '%s'", str.c_str());

    return DataTypeWrapper<float>();
}

} // namespace mirheo
