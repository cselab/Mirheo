#pragma once

#include <core/datatypes.h>

#define TYPE_TABLE(OP)                           \
    OP(float)                                    \
    OP(double)                                   \
    OP(int)                                      \
    OP(float3)                                   \
    OP(Particle)

#define TOKENIZE(ctype) _##ctype##_




enum class DataType
{
#define MAKE_ENUM(ctype) TOKENIZE(ctype),
    TYPE_TABLE(MAKE_ENUM)
#undef MAKE_ENUM
    None
};




template<typename T> DataType typeTokenize() { return DataType::None; }

#define MAKE_TOKENIZE_FUNCTIONS(ctype) \
    template<> inline DataType typeTokenize<ctype>() {return DataType::TOKENIZE(ctype);}

TYPE_TABLE(MAKE_TOKENIZE_FUNCTIONS)

#undef MAKE_TOKENIZE_FUNCTIONS



/* usage:

DataType dataType = ...;

switch(dataType) {
#define SWITCH_ENTRY(ctype)                     \
    case DataType::TOKENIZE(ctype):             \
        CALL_TEMPLATED_FUNCTION<ctype>();       \
        break;                                  \

    TYPE_TABLE(SWITCH_ENTRY)
        
#undef SWITCH_ENTRY
};
*/
