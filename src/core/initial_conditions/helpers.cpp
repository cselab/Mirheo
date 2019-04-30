#include <functional>
#include <random>

#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/common.h>

#include "helpers.h"

using PositionFilter = std::function<bool(float3)>;

static long genSeed(const MPI_Comm& comm, std::string name)
{
    int rank;
    std::hash<std::string> nameHash;

    MPI_Check( MPI_Comm_rank(comm, &rank) );
    return rank + nameHash(name);
}

static Particle genParticle(float3 h, int i, int j, int k, const DomainInfo& domain,
                            std::uniform_real_distribution<float>& udistr, std::mt19937& gen)
{
    Particle p;
    p.r.x = i*h.x - 0.5*domain.localSize.x + udistr(gen);
    p.r.y = j*h.y - 0.5*domain.localSize.y + udistr(gen);
    p.r.z = k*h.z - 0.5*domain.localSize.z + udistr(gen);

    p.u.x = 0.0f * (udistr(gen) - 0.5);
    p.u.y = 0.0f * (udistr(gen) - 0.5);
    p.u.z = 0.0f * (udistr(gen) - 0.5);

    return p;
}

void addUniformParticles(float density, const MPI_Comm& comm, ParticleVector *pv, PositionFilter filterIn, cudaStream_t stream)
{
    auto domain = pv->state->domain;

    int3 ncells = make_int3( ceilf(domain.localSize) );
    float3 h    = domain.localSize / make_float3(ncells);

    int wholeInCell = floor(density);
    float fracInCell = density - wholeInCell;

    auto seed = genSeed(comm, pv->name);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> udistr(0, 1);

    double3 avgMomentum{0,0,0};
    int mycount = 0;

    for (int i = 0; i < ncells.x; i++) {
        for (int j = 0; j < ncells.y; j++) {
            for (int k = 0; k < ncells.z; k++) {

                int nparts = wholeInCell;
                if (udistr(gen) < fracInCell) nparts++;

                for (int p = 0; p < nparts; p++) {

                    auto part = genParticle(h, i, j, k, domain, udistr, gen);

                    if (! filterIn(domain.local2global(part.r)))
                        continue;

                    pv->local()->resize(mycount+1,  stream);
                    auto pos = pv->local()->positions ().hostPtr();
                    auto vel = pv->local()->velocities().hostPtr();

                    pos[mycount] = part.r2Float4();
                    vel[mycount] = part.u2Float4();

                    avgMomentum.x += part.u.x;
                    avgMomentum.y += part.u.y;
                    avgMomentum.z += part.u.z;

                    mycount++;
                }
            }
        }
    }

    avgMomentum.x /= mycount;
    avgMomentum.y /= mycount;
    avgMomentum.z /= mycount;

    for (auto& vel : pv->local()->velocities()) {
        vel.x -= avgMomentum.x;
        vel.y -= avgMomentum.y;
        vel.z -= avgMomentum.z;
    }

    pv->local()->positions() .uploadToDevice(stream);
    pv->local()->velocities().uploadToDevice(stream);
    pv->local()->computeGlobalIds(comm, stream);
    pv->local()->dataPerParticle.getData<float4>(ChannelNames::oldPositions)->copy(pv->local()->positions(), stream);

    debug2("Generated %d %s particles", pv->local()->size(), pv->name.c_str());
}
