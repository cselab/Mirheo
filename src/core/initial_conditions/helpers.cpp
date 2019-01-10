#include <functional>
#include <random>

#include <core/pvs/particle_vector.h>
#include <core/logger.h>

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
                    part.i1 = mycount;

                    if (! filterIn(domain.local2global(part.r)))
                        continue;

                    pv->local()->resize(mycount+1,  stream);
                    auto cooPtr = pv->local()->coosvels.hostPtr();

                    cooPtr[mycount] = part;

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

    auto cooPtr = pv->local()->coosvels.hostPtr();

    for (auto& part : pv->local()->coosvels) {
        part.u.x -= avgMomentum.x;
        part.u.y -= avgMomentum.y;
        part.u.z -= avgMomentum.z;
    }

    int totalCount = 0; // TODO: int64!
    MPI_Check( MPI_Exscan(&mycount, &totalCount, 1, MPI_INT, MPI_SUM, comm) );
    for (auto& part : pv->local()->coosvels)
        part.i1 += totalCount;

    pv->local()->coosvels.uploadToDevice(stream);
    pv->local()->extraPerParticle.getData<Particle>("old_particles")->copy(pv->local()->coosvels, stream);

    debug2("Generated %d %s particles", pv->local()->size(), pv->name.c_str());
}
