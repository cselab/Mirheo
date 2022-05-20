// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "chain.h"

#include <mirheo/core/pvs/chain_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

ChainIC::ChainIC(std::vector<real3> positions, std::vector<real3> orientations, real length) :
    positions_(std::move(positions)),
    orientations_(std::move(orientations)),
    length_(length)
{
    if (positions.size() != orientations.size())
        die("ChainIC: Positions and orientations must have the same number of elements.");
}

ChainIC::~ChainIC() = default;


static std::vector<int> getIdsLocalChains(const std::vector<real3>& positions,
                                          const DomainInfo& domain)
{
    std::vector<int> ids;
    for (size_t i = 0; i < positions.size(); ++i)
    {
        const real3 com = positions[i];
        if (domain.inSubDomain(com))
            ids.push_back(static_cast<int>(i));
    }
    return ids;
}

void ChainIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    auto ov = dynamic_cast<ChainVector*>(pv);
    const auto domain = pv->getState()->domain;

    if (ov == nullptr)
        die("Chains can only be generated out of chain object vectors");

    LocalObjectVector *lov = ov->local();

    const auto localIds = getIdsLocalChains(positions_, domain);

    const int nObjsLocal = static_cast<int>(localIds.size());
    const int objSize = ov->getObjectSize();

    lov->resize_anew(nObjsLocal * objSize);
    auto& pos = ov->local()->positions();
    auto& vel = ov->local()->velocities();

    const real h = length_ / (static_cast<real>(objSize - 1));

    for (size_t objId = 0; objId < localIds.size(); ++objId)
    {
        const int srcId = localIds[objId];
        const real3 com = domain.global2local(positions_[srcId]);
        const real3 u = normalize(orientations_[srcId]);

        for (int i = 0; i < objSize; ++i)
        {
            const real3 r = com + (h * static_cast<real>(i) - 0.5_r * length_) * u;
            const Particle p {{r.x, r.y, r.z, 0._r}, make_real4(0._r)};

            const size_t dstPid = objId * objSize + i;

            pos[dstPid] = p.r2Real4();
            vel[dstPid] = p.u2Real4();
        }
    }

    lov->positions() .uploadToDevice(stream);
    lov->velocities().uploadToDevice(stream);
    lov->computeGlobalIds(comm, stream);
    lov->dataPerParticle.getData<real4>(channel_names::oldPositions)->copy(ov->local()->positions(), stream);

    info("Initialized %d '%s' chains", nObjsLocal, ov->getCName());
}

} // namespace mirheo
