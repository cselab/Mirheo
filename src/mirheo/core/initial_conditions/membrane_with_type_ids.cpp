#include "membrane_with_type_ids.h"

#include <mirheo/core/pvs/membrane_vector.h>

MembraneWithTypeIdsIC::MembraneWithTypeIdsIC(const std::vector<ComQ>& com_q,
                                             const std::vector<int>& typeIds,
                                             real globalScale) :
    MembraneIC(com_q, globalScale),
    typeIds(typeIds)
{}

MembraneWithTypeIdsIC::~MembraneWithTypeIdsIC() = default;

void MembraneWithTypeIdsIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    MembraneIC::exec(comm, pv, stream);

    auto ov = static_cast<MembraneVector*>(pv);
    const auto domain = pv->state->domain;
    LocalObjectVector *lov = ov->local();

    const auto map = createMap(domain);
    const int nObjsLocal = map.size();

    ov->requireDataPerObject<int>(ChannelNames::membraneTypeId, DataManager::PersistenceMode::Active);

    auto typeIdsBuff = lov->dataPerObject.getData<int>(ChannelNames::membraneTypeId);

    typeIdsBuff->resize_anew(nObjsLocal);

    for (size_t objId = 0; objId < map.size(); ++objId)
    {
        const int srcId = map[objId];
        typeIdsBuff[objId] = typeIds[srcId];
    }
    typeIdsBuff->uploadToDevice(stream);

    info("Initialized %d '%s' membrane type ids", nObjsLocal, ov->name.c_str());
}

