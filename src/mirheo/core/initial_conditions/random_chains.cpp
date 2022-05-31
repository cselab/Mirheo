// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "random_chains.h"

#include <mirheo/core/pvs/chain_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/helper_math.h>

#include <cmath>
#include <random>

namespace mirheo
{

RandomChainsIC::RandomChainsIC(std::vector<real3> positions,
                               real length) :
    positions_(std::move(positions)),
    length_(length)
{}

RandomChainsIC::~RandomChainsIC() = default;


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

static real3 randomPointUnitSphere(std::mt19937& gen)
{
    std::uniform_real_distribution<real> u(0.0_r, 1.0_r);

    const real theta = 2 * static_cast<real>(M_PI) * u(gen);
    const real phi = std::acos(1.0_r - 2.0_r * u(gen));

    return {std::sin(phi) * std::cos(theta),
            std::sin(phi) * std::sin(theta),
            std::cos(phi)};
}

std::vector<real3> generateChain(int numBeads, real totalLength, std::mt19937& gen)
{
    const real h = totalLength / static_cast<real>(numBeads - 1);

    std::vector<real3> positions(numBeads);
    real3 com = make_real3(0.0_r);
    positions[0] = com;

    for (int i = 1; i < numBeads; ++i)
    {
        positions[i] = positions[i-1] + h * randomPointUnitSphere(gen);
        com += positions[i];
    }

    com /= static_cast<real>(numBeads);

    for (auto& r : positions)
        r -= com;

    return positions;
}

void RandomChainsIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
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

    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    const long seed = rank * 17 + 42;

    std::mt19937 gen(seed);

    for (size_t objId = 0; objId < localIds.size(); ++objId)
    {
        const int srcId = localIds[objId];
        const real3 com = domain.global2local(positions_[srcId]);
        const auto positions = generateChain(objSize, length_, gen);

        for (int i = 0; i < objSize; ++i)
        {
            const real3 r = com + positions[i];
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
