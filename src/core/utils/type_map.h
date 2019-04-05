#pragma once

#include <string>
#include <core/datatypes.h>
#include <extern/variant/include/mpark/variant.hpp>

#define TYPE_TABLE__(OP, SEP)                   \
    OP(int)          SEP                        \
    OP(float)        SEP                        \
    OP(float2)       SEP                        \
    OP(float3)       SEP                        \
    OP(float4)       SEP                        \
    OP(double)       SEP                        \
    OP(double3)      SEP                        \
    OP(double4)      SEP                        \
    OP(Stress)       SEP                        \
    OP(RigidMotion)  SEP                        \
    OP(COMandExtent) SEP                        \
    OP(Particle)


#define TYPE_TABLE(OP) TYPE_TABLE__(OP, )
#define COMMA ,
#define TYPE_TABLE_COMMA(OP) TYPE_TABLE__(OP, COMMA)


template<class T>
struct DataTypeWrapper {using type = T;};

using TypeDescriptor = mpark::variant<
#define MAKE_WRAPPER(a) DataTypeWrapper<a>
    TYPE_TABLE_COMMA(MAKE_WRAPPER)
#undef MAKE_WRAPPER
    >;


std::string typeDescriptorToString(const TypeDescriptor& desc);
TypeDescriptor stringToTypeDescriptor(const std::string& str);




// TODO: remove from here


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

template<typename T> DataType inline typeTokenize() { return DataType::DATATYPE_NONE; }

#define MAKE_TOKENIZE_FUNCTIONS(ctype) \
    template<> inline DataType typeTokenize<ctype>() {return DataType::TOKENIZE(ctype);}

TYPE_TABLE(MAKE_TOKENIZE_FUNCTIONS)

#undef MAKE_TOKENIZE_FUNCTIONS

