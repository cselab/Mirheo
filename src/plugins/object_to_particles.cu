#include "object_to_particles.h"

#include <core/pvs/views/ov.h>
#include <core/utils/kernel_launch.h>
#include <core/simulation.h>

// NOTE: Work in progress! Currently, instead of transforming the object to
// particles, the objects are simply deleted.

// TODO: Some mechanism for arbitrary types. Something like walls have now, but
// then without duplication of code here and there?

namespace ObjectToParticles {

__global__ void markObjects(OVview view, ObjectDeleterHandler deleter, real4 plane)
{
    int oid = blockIdx.x * blockDim.x + threadIdx.x;
    if (oid >= view.nObjects) return;

    auto &com = view.comAndExtents[oid].com;
    real tmp = plane.x * com.x + plane.y * com.y + plane.z * com.z + plane.w;
    if (tmp > 0.f)
        deleter.mark(oid);
}

} // namespace ObjectToParticles


ObjectToParticlesPlugin::ObjectToParticlesPlugin(
        const MirState *state, std::string name,
        std::string ovName, std::string pvName, real4 plane) :
    SimulationPlugin(state, name),
    ovName(ovName),
    pvName(pvName),
    plane(state->domain.global2localPlane(plane))
{}

ObjectToParticlesPlugin::~ObjectToParticlesPlugin() = default;

void ObjectToParticlesPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    ov = simulation->getOVbyNameOrDie(ovName);
    pv = simulation->getPVbyNameOrDie(pvName);
}

void ObjectToParticlesPlugin::afterIntegration(cudaStream_t stream)
{
    const int nthreads = 128;

    ov->findExtentAndCOM(stream, ParticleVectorLocality::Local);
    if (ov->local()->nObjects || ov->halo()->nObjects)
        info("ObjectVector local = %d   halo = %d", ov->local()->nObjects, ov->halo()->nObjects);
    LocalObjectVector *lov = ov->local();

    deleter.update(lov, stream);

    SAFE_KERNEL_LAUNCH(
        ObjectToParticles::markObjects,
        getNblocks(lov->nObjects, nthreads), nthreads, 0, stream,
        OVview(ov, lov), deleter.handler(), plane);

    // Delete objects and move their particles to `pv`.
    // deleter.deleteObjects(lov, stream, pv->local());
    deleter.deleteObjects(lov, stream);
}
