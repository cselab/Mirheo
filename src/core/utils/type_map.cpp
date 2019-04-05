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
}




std::string dataTypeToString(DataType dataType)
{
#define SWITCH_ENTRY(ctype) case DataType::TOKENIZE(ctype): return #ctype;
#define GET_STR(a) #a
    
    switch (dataType) {
        TYPE_TABLE(SWITCH_ENTRY);
    default: return GET_STR(DATATYPE_NONE);
    };

#undef SWITCH_ENTRY
#undef GET_STR
}

DataType stringToDataType(std::string str)
{
#define IF_ENTRY(ctype) if (str == #ctype) return DataType::TOKENIZE(ctype);

    TYPE_TABLE(IF_ENTRY);

#undef IF_ENTRY

    return DataType::DATATYPE_NONE;
}
