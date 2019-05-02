#pragma once

#include <core/celllist.h>

/**
 * map threads to cell id inside a given face
 * @param gid: global thread id
 * @param faceId: face index, from 0 to 5
 * @param cinfo: cells infos
 * @param cid: returned cell id corresponding to the thread and face
 * @param dx: returned direction (-1, 0 or 1) along x
 * @param dy: returned direction (-1, 0 or 1) along y
 * @param dz: returned direction (-1, 0 or 1) along z
 * @return true if the thread is participating, false otherwise
 */
__device__ inline bool distributeThreadsToFaceCell(int& cid, int& dx, int& dy, int& dz, int gid, int faceId, CellListInfo cinfo)
{
    const int3 ncells = cinfo.ncells;

    bool valid = true;

    if (faceId <= 1)  // x
    {
        if (gid >= ncells.y * ncells.z) valid = false;
        dx = faceId * (ncells.x - 1);
        dy = gid % ncells.y;
        dz = gid / ncells.y;
    }
    else if (faceId <= 3)  // y
    {
        if (gid >= ncells.x * ncells.z) valid = false;
        dx = gid % ncells.x;
        dy = (faceId - 2) * (ncells.y - 1);
        dz = gid / ncells.x;
    }
    else   // z
    {
        if (gid >= ncells.x * ncells.y) valid = false;
        dx = gid % ncells.x;
        dy = gid / ncells.x;
        dz = (faceId - 4) * (ncells.z - 1);
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

    // Exclude cells already covered by other faceIds
    if ( (faceId == 0 || faceId == 1) && (dz != 0 || dy != 0) ) valid = false;
    if ( (faceId == 2 || faceId == 3) && (dz != 0) ) valid = false;

    return valid;
}
