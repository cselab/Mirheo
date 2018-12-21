#include <core/logger.h>

#include "domain.h"

DomainInfo createDomainInfo(MPI_Comm cartComm, float3 globalSize)
{
    DomainInfo domain;
    int ranks[3], periods[3], coords[3];

    MPI_Check(MPI_Cart_get(cartComm, 3, ranks, periods, coords));

    int3 nranks3D = {ranks[0], ranks[1], ranks[2]};
    int3 rank3D = {coords[0], coords[1], coords[2]};

    domain.globalSize = globalSize;
    domain.localSize = domain.globalSize / make_float3(nranks3D);
    domain.globalStart = domain.localSize * make_float3(rank3D);

    return domain;
}
