#pragma once

#include <string>
#include <core/datatypes.h>


#define TYPE_TABLE_ADDITIONABLE(OP)              \
    OP(float)                                    \
    OP(double)                                   \
    OP(int)                                      \
    OP(float3)                                   \
    OP(float4)                                   \
    OP(double3)                                  \
    OP(double4)                                  \
    OP(Stress)

#define TYPE_TABLE(OP)                           \
    TYPE_TABLE_ADDITIONABLE(OP)                  \
    OP(Particle)

#define DATATYPE_NONE None

#define TOKENIZE(ctype) _##ctype##_


enum class DataType
{
#define MAKE_ENUM(ctype) TOKENIZE(ctype),
    TYPE_TABLE(MAKE_ENUM)
#undef MAKE_ENUM
    DATATYPE_NONE
};


std::string dataTypeToString(DataType dataType);
DataType stringToDataType(std::string str);

size_t dataTypeToByteSize(DataType dataType);

template<typename T> DataType inline typeTokenize() { return DataType::DATATYPE_NONE; }

#define MAKE_TOKENIZE_FUNCTIONS(ctype) \
    template<> inline DataType typeTokenize<ctype>() {return DataType::TOKENIZE(ctype);}

TYPE_TABLE(MAKE_TOKENIZE_FUNCTIONS)

#undef MAKE_TOKENIZE_FUNCTIONS

