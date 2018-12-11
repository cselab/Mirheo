#pragma once

#define TYPE_TABLE(OP)                           \
    OP(float)                                    \
    OP(double)                                   \
    OP(int)                                      \
    OP(float4)

#define TOKENIFY(ctype) _##ctype##_




enum class DataType
{
#define MAKE_ENUM(ctype) TOKENIFY(ctype),
    TYPE_TABLE(MAKE_ENUM)
#undef MAKE_ENUM
};




template<typename T> DataType typeTokenify();

#define MAKE_TOKENIFY_FUNCTIONS(ctype) \
    template<> inline DataType typeTokenify<ctype>() {return DataType::TOKENIFY(ctype);}

TYPE_TABLE(MAKE_TOKENIFY_FUNCTIONS)

#undef MAKE_TOKENIFY_FUNCTIONS
