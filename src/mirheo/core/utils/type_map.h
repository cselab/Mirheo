#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/rigid/rigid_motion.h>
#include <mirheo/core/utils/cuda_variant.h>

#include <extern/variant/include/mpark/variant.hpp>

#include <string>

namespace mirheo
{

#define MIRHEO_TYPE_TABLE__(OP, SEP)            \
    OP(int)          SEP                        \
    OP(int64_t)      SEP                        \
    OP(float)        SEP                        \
    OP(float2)       SEP                        \
    OP(float3)       SEP                        \
    OP(float4)       SEP                        \
    OP(double)       SEP                        \
    OP(double2)      SEP                        \
    OP(double3)      SEP                        \
    OP(double4)      SEP                        \
    OP(Stress)       SEP                        \
    OP(RigidMotion)  SEP                        \
    OP(COMandExtent) SEP                        \
    OP(Force)


#define MIRHEO_TYPE_TABLE(OP) MIRHEO_TYPE_TABLE__(OP, )
#define COMMA ,
#define MIRHEO_TYPE_TABLE_COMMA(OP) MIRHEO_TYPE_TABLE__(OP, COMMA)


template<class T>
struct DataTypeWrapper {using type = T;};

using TypeDescriptor = mpark::variant<
#define MAKE_WRAPPER(a) DataTypeWrapper<a>
    MIRHEO_TYPE_TABLE_COMMA(MAKE_WRAPPER)
#undef MAKE_WRAPPER
    >;

using CudaVarPtr = cuda_variant::variant<
#define MAKE_WRAPPER(a) a*
    MIRHEO_TYPE_TABLE_COMMA(MAKE_WRAPPER)
#undef MAKE_WRAPPER
    >;

std::string typeDescriptorToString(const TypeDescriptor& desc);
TypeDescriptor stringToTypeDescriptor(const std::string& str);

} // namespace mirheo
