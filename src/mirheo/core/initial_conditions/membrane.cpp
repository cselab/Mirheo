#include "membrane.h"

#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/quaternion.h>

namespace mirheo
{

MembraneIC::MembraneIC(const std::vector<ComQ>& com_q, real globalScale) :
    com_q(com_q),
    globalScale(globalScale)
{}

MembraneIC::~MembraneIC() = default;

void MembraneIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    auto ov = dynamic_cast<MembraneVector*>(pv);
    const auto domain = pv->state->domain;
    
    if (ov == nullptr)
        die("RBCs can only be generated out of rbc object vectors");

    LocalObjectVector *lov = ov->local();

    const auto map = createMap(domain);
    
    const int nObjsLocal = static_cast<int>(map.size());
    const int nVerticesPerObject = ov->mesh->getNvertices();
    
    lov->resize_anew(nObjsLocal * nVerticesPerObject);
    auto& pos = ov->local()->positions();
    auto& vel = ov->local()->velocities();

    for (size_t objId = 0; objId < map.size(); ++objId)
    {
        const int srcId = map[objId];
        const real3 com = domain.global2local(com_q[srcId].r);
        const auto q = Quaternion<real>::createFromComponents(normalize(com_q[srcId].q));

        for (int i = 0; i < nVerticesPerObject; ++i)
        {
            const real3 dr0 = make_real3( ov->mesh->vertexCoordinates[i] * globalScale);
            const real3 r = com + q.rotate(dr0);
            const Particle p {{r.x, r.y, r.z, 0._r}, make_real4(0._r)};

            const size_t dstPid = objId * nVerticesPerObject + i;
            
            pos[dstPid] = p.r2Real4();
            vel[dstPid] = p.u2Real4();
        }
    }

    lov->positions() .uploadToDevice(stream);
    lov->velocities().uploadToDevice(stream);
    lov->computeGlobalIds(comm, stream);
    lov->dataPerParticle.getData<real4>(ChannelNames::oldPositions)->copy(ov->local()->positions(), stream);

    info("Initialized %d '%s' membranes", nObjsLocal, ov->name.c_str());
}

std::vector<int> MembraneIC::createMap(DomainInfo domain) const
{
    std::vector<int> map;
    for (size_t i = 0; i < com_q.size(); ++i)
    {
        const real3 com = com_q[i].r;
        if (domain.inSubDomain(com))
            map.push_back(static_cast<int>(i));
    }
    return map;
}

} // namespace mirheo
