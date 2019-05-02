#pragma once

#include <core/celllist.h>

__device__ inline bool isValidCell(int& cid, int& dx, int& dy, int& dz, int gid, int variant, CellListInfo cinfo)
{
    const int3 ncells = cinfo.ncells;

    bool valid = true;

    if (variant <= 1)  // x
    {
        if (gid >= ncells.y * ncells.z) valid = false;
        dx = variant * (ncells.x - 1);
        dy = gid % ncells.y;
        dz = gid / ncells.y;
    }
    else if (variant <= 3)  // y
    {
        if (gid >= ncells.x * ncells.z) valid = false;
        dx = gid % ncells.x;
        dy = (variant - 2) * (ncells.y - 1);
        dz = gid / ncells.x;
    }
    else   // z
    {
        if (gid >= ncells.x * ncells.y) valid = false;
        dx = gid % ncells.x;
        dy = gid / ncells.x;
        dz = (variant - 4) * (ncells.z - 1);
    }

    cid = cinfo.encode(dx, dy, dz);

    valid &= cid < cinfo.totcells;

    // Find side directions
    if (dx == 0) dx = -1;
    else if (dx == ncells.x-1) dx = 1;
    else dx = 0;

    if (dy == 0) dy = -1;
    else if (dy == ncells.y-1) dy = 1;
    else dy = 0;

    if (dz == 0) dz = -1;
    else if (dz == ncells.z-1) dz = 1;
    else dz = 0;

    // Exclude cells already covered by other variants
    if ( (variant == 0 || variant == 1) && (dz != 0 || dy != 0) ) valid = false;
    if ( (variant == 2 || variant == 3) && (dz != 0) ) valid = false;

    return valid;
}
