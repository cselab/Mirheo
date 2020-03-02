#include "membrane.h"

#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/quaternion.h>

namespace mirheo
{

MembraneIC::MembraneIC(const std::vector<ComQ>& comQ, real globalScale) :
    comQ_(comQ),
    globalScale_(globalScale)
{}

MembraneIC::~MembraneIC() = default;

void MembraneIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    auto ov = dynamic_cast<MembraneVector*>(pv);
    const auto domain = pv->getState()->domain;
    
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
        const real3 com = domain.global2local(comQ_[srcId].r);
        const real4 q4 = comQ_[srcId].q;
        if (dot(q4, q4) == 0.0)
            die("Quaternion must be non-zero.");
        const auto q = Quaternion<real>::createFromComponents(normalize(q4));

        for (int i = 0; i < nVerticesPerObject; ++i)
        {
            const real3 dr0 = make_real3( ov->mesh->getVertices()[i] * globalScale_);
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

    info("Initialized %d '%s' membranes", nObjsLocal, ov->getCName());
}

std::vector<int> MembraneIC::createMap(DomainInfo domain) const
{
    std::vector<int> map;
    for (size_t i = 0; i < comQ_.size(); ++i)
    {
        const real3 com = comQ_[i].r;
        if (domain.inSubDomain(com))
            map.push_back(static_cast<int>(i));
    }
    return map;
}

} // namespace mirheo
