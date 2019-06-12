#pragma once


#include <core/datatypes.h>
#include <core/utils/cuda_variant.h>

#include <extern/variant/include/mpark/variant.hpp>

#include <string>

#define TYPE_TABLE__(OP, SEP)                   \
    OP(int)          SEP                        \
    OP(int64_t)      SEP                        \
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
    OP(Force)


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

using CudaVarPtr = cuda_variant::variant<
#define MAKE_WRAPPER(a) a*
    TYPE_TABLE_COMMA(MAKE_WRAPPER)
#undef MAKE_WRAPPER
    >;

std::string typeDescriptorToString(const TypeDescriptor& desc);
TypeDescriptor stringToTypeDescriptor(const std::string& str);
