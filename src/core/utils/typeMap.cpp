#include "typeMap.h"

std::string dataTypeToString(DataType dataType)
{
    switch (dataType) {

#define SWITCH_ENTRY(ctype) case DataType::TOKENIZE(ctype): return #ctype;

        TYPE_TABLE(SWITCH_ENTRY);

#undef SWITCH_ENTRY

    default: return #DATATYPE_NONE;
    };
}

DataType stringToDataType(std::string str)
{
#define IF_ENTRY(ctype) if (str == #ctype) return DataType::TOKENIZE(ctype);

    TYPE_TABLE(IF_ENTRY);

#undef SWITCH_ENTRY

    return DataType::DATATYPE_NONE;
}

