#include "type_map.h"

struct VisitorToStr
{
#define TYPE2STR(Type) std::string operator()(const DataTypeWrapper<Type>&) const {return #Type ;}

    TYPE_TABLE(TYPE2STR)
    
#undef TYPE2STR
};

std::string typeDescriptorToString(const TypeDescriptor& desc)
{
    return mpark::visit(VisitorToStr(), desc);
}

TypeDescriptor stringToTypeDescriptor(const std::string& str)
{
#define IF_ENTRY(Type) if (str == #Type) return { DataTypeWrapper<Type>() };

    TYPE_TABLE(IF_ENTRY);

#undef IF_ENTRY

    die("Unrecognized type '%s'", str);

    return DataTypeWrapper<float>();
}

