// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "domain.h"
#include <mirheo/core/logger.h>

namespace mirheo
{

DomainInfo createDomainInfo(MPI_Comm cartComm, real3 globalSize)
{
    DomainInfo domain;
    int ranks[3], periods[3], coords[3];

    MPI_Check(MPI_Cart_get(cartComm, 3, ranks, periods, coords));

    int3 nranks3D = {ranks[0], ranks[1], ranks[2]};
    int3 rank3D = {coords[0], coords[1], coords[2]};

    domain.globalSize = globalSize;
    domain.localSize = domain.globalSize / make_real3(nranks3D);
    domain.globalStart = domain.localSize * make_real3(rank3D);

    const real maxSide = std::max(globalSize.x, std::max(globalSize.y, globalSize.z));
    if (maxSide > std::fabs((real)Real3_int::mark_val)) {
        die("domain size (%g %g %g) too large, side length must be smaller than %g",
            globalSize.x, globalSize.y, globalSize.z, Real3_int::mark_val);
    }

    return domain;
}

} // namespace mirheo
