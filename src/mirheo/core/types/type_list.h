#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/rigid/rigid_motion.h>

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

} // namespace mirheo
