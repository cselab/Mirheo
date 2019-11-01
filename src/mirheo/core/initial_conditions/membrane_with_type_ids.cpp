#include "membrane_with_type_ids.h"

#include <mirheo/core/pvs/membrane_vector.h>

namespace mirheo
{

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
    const auto map = createMap(domain);
    const int nObjsLocal = map.size();

    ov->requireDataPerObject<int>(ChannelNames::membraneTypeId, DataManager::PersistenceMode::Active);
    LocalObjectVector *lov = ov->local();

    auto& typeIdsBuff = *lov->dataPerObject.getData<int>(ChannelNames::membraneTypeId);

    for (int objId = 0; objId < nObjsLocal; ++objId)
    {
        const int srcId = map[objId];
        typeIdsBuff[objId] = typeIds[srcId];
    }
    typeIdsBuff.uploadToDevice(stream);

    info("Initialized %d '%s' membrane type ids", nObjsLocal, ov->name.c_str());
}

} // namespace mirheo
