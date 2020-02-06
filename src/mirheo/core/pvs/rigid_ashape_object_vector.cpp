#include "rigid_ashape_object_vector.h"

#include <mirheo/core/analytical_shapes/api.h>

namespace mirheo
{

template <class Shape> RigidShapedObjectVector<Shape>::
RigidShapedObjectVector(const MirState *state, const std::string& name, real mass, int objSize,
                        Shape shape, int nObjects) :
    RigidObjectVector(state, name, mass,
                      shape.inertiaTensor(mass * static_cast<real>(objSize)),
                      objSize,
                      std::make_shared<Mesh>(),
                      nObjects),
    shape_(shape)
{}

template <class Shape> RigidShapedObjectVector<Shape>::
RigidShapedObjectVector(const MirState *state, const std::string& name, real mass, int objSize,
                        Shape shape, std::shared_ptr<Mesh> mesh_, int nObjects) :
    RigidObjectVector(state, name, mass,
                      shape.inertiaTensor(mass * static_cast<real>(objSize)),
                      objSize, std::move(mesh_), nObjects),
    shape_(shape)
{}
        
template <class Shape> RigidShapedObjectVector<Shape>::
~RigidShapedObjectVector() = default;


#define INSTANTIATE(Shape) template class RigidShapedObjectVector<Shape>;
ASHAPE_TABLE(INSTANTIATE)
#undef INSTANTIATE

} // namespace mirheo
