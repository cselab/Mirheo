#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <vector_types.h>

namespace mirheo
{

namespace fragment_mapping
{

inline __HD__ int getDirx(int id) { return (id     + 2) % 3 - 1; }
inline __HD__ int getDiry(int id) { return (id / 3 + 2) % 3 - 1; }
inline __HD__ int getDirz(int id) { return (id / 9 + 2) % 3 - 1; }

inline __HD__ int3 getDir(int id)
{
    return {getDirx(id), getDiry(id), getDirz(id)};
}

inline __HD__ int getId(int dx, int dy, int dz)
{
    return    ((dx + 2) % 3)
        + 3 * ((dy + 2) % 3)
        + 9 * ((dz + 2) % 3);
}

inline __HD__ int getId(int3 dir)
{
    return getId(dir.x, dir.y, dir.z);
}

constexpr int numFragments = 27;
constexpr int bulkId       = 26;

} // namespace fragment_mapping
} // namespace mirheo
