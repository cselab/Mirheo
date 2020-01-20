#include <functional>
#include <random>

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/common.h>

#include "helpers.h"

namespace mirheo
{

using PositionFilter = std::function<bool(real3)>;

static long genSeed(const MPI_Comm& comm, const std::string& name)
{
    int rank;
    std::hash<std::string> nameHash;

    MPI_Check( MPI_Comm_rank(comm, &rank) );
    return rank + nameHash(name);
}

static Particle genParticle(real3 h, int i, int j, int k, const DomainInfo& domain,
                            std::uniform_real_distribution<float>& udistr, std::mt19937& gen)
{
    Particle p;
    p.r.x = static_cast<real>(i)*h.x - 0.5_r * domain.localSize.x + udistr(gen);
    p.r.y = static_cast<real>(j)*h.y - 0.5_r * domain.localSize.y + udistr(gen);
    p.r.z = static_cast<real>(k)*h.z - 0.5_r * domain.localSize.z + udistr(gen);

    p.u.x = 0.0_r * (udistr(gen) - 0.5_r);
    p.u.y = 0.0_r * (udistr(gen) - 0.5_r);
    p.u.z = 0.0_r * (udistr(gen) - 0.5_r);

    return p;
}

void addUniformParticles(real numberDensity, const MPI_Comm& comm, ParticleVector *pv, PositionFilter filterIn, cudaStream_t stream)
{
    const auto domain = pv->getState()->domain;

    const int3 ncells     = make_int3( math::ceil(domain.localSize) );
    const real3 h         = domain.localSize / make_real3(ncells);
    const real cellVolume = h.x * h.y * h.z;

    const real numPartsPerCell = cellVolume * numberDensity;

    const int wholeInCell = static_cast<int>(math::floor(numPartsPerCell));
    const real fracInCell = numPartsPerCell - static_cast<real>(wholeInCell);

    const auto seed = genSeed(comm, pv->name);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> udistr(0, 1); // use float to get the same refs for tests

    double3 avgMomentum {0,0,0};
    int mycount {0};

    std::vector<float4> pos, vel;
    pos.reserve(ncells.x * ncells.y * ncells.z * static_cast<int>(math::ceil(numPartsPerCell)));
    vel.reserve(ncells.x * ncells.y * ncells.z * static_cast<int>(math::ceil(numPartsPerCell)));
    
    for (int i = 0; i < ncells.x; ++i) {
        for (int j = 0; j < ncells.y; ++j) {
            for (int k = 0; k < ncells.z; ++k) {

                int nparts = wholeInCell;
                if (udistr(gen) < fracInCell)
                    ++nparts;

                for (int p = 0; p < nparts; ++p)
                {
                    const Particle part = genParticle(h, i, j, k, domain, udistr, gen);

                    if (! filterIn(domain.local2global(part.r)))
                        continue;

                    pos.push_back(part.r2Real4());
                    vel.push_back(part.u2Real4());

                    avgMomentum.x += part.u.x;
                    avgMomentum.y += part.u.y;
                    avgMomentum.z += part.u.z;

                    mycount++;
                }
            }
        }
    }

    pv->local()->resize(mycount, stream);
    std::copy(pos.begin(), pos.end(), pv->local()->positions ().begin());
    std::copy(vel.begin(), vel.end(), pv->local()->velocities().begin());
    
    avgMomentum.x /= mycount;
    avgMomentum.y /= mycount;
    avgMomentum.z /= mycount;

    for (auto& vel : pv->local()->velocities())
    {
        vel.x -= static_cast<real>(avgMomentum.x);
        vel.y -= static_cast<real>(avgMomentum.y);
        vel.z -= static_cast<real>(avgMomentum.z);
    }

    pv->local()->positions() .uploadToDevice(stream);
    pv->local()->velocities().uploadToDevice(stream);
    pv->local()->computeGlobalIds(comm, stream);
    pv->local()->dataPerParticle.getData<real4>(ChannelNames::oldPositions)->copy(pv->local()->positions(), stream);

    debug2("Generated %d %s particles", pv->local()->size(), pv->name.c_str());
}

} // namespace mirheo
