// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "membrane_with_type_ids.h"

#include <mirheo/core/pvs/membrane_vector.h>

namespace mirheo
{

MembraneWithTypeIdsIC::MembraneWithTypeIdsIC(const std::vector<ComQ>& comQ,
                                             const std::vector<int>& typeIds,
                                             real globalScale) :
    MembraneIC(comQ, globalScale),
    typeIds_(typeIds)
{}

MembraneWithTypeIdsIC::~MembraneWithTypeIdsIC() = default;

void MembraneWithTypeIdsIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    MembraneIC::exec(comm, pv, stream);

    auto ov = static_cast<MembraneVector*>(pv);
    const auto domain = pv->getState()->domain;
    const auto map = createMap(domain);
    const int nObjsLocal = static_cast<int>(map.size());

    ov->requireDataPerObject<int>(channel_names::membraneTypeId, DataManager::PersistenceMode::Active);
    LocalObjectVector *lov = ov->local();

    auto& typeIdsBuff = *lov->dataPerObject.getData<int>(channel_names::membraneTypeId);

    for (int objId = 0; objId < nObjsLocal; ++objId)
    {
        const int srcId = map[objId];
        typeIdsBuff[objId] = typeIds_[srcId];
    }
    typeIdsBuff.uploadToDevice(stream);

    info("Initialized %d '%s' membrane type ids", nObjsLocal, ov->getCName());
}

} // namespace mirheo
