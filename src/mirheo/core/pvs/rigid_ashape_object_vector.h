// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "rigid_object_vector.h"

namespace mirheo
{

/** \brief RigidObjectVector with analytic shape instead of triangle mesh.
    \tparam Shape The analytic shape that represents the object surface in its frame of reference
 */
template <class Shape>
class RigidShapedObjectVector : public RigidObjectVector
{
public:
    /** Construct a RigidShapedObjectVector
        \param [in] state The simulation state
        \param [in] name Name of the pv
        \param [in] mass Mass of one frozen particle
        \param [in] objSize Number of particles per object
        \param [in] shape The shape that represents the surface of the object
        \param [in] nObjects Number of objects
    */
    RigidShapedObjectVector(const MirState *state, const std::string& name, real mass, int objSize,
                            Shape shape, int nObjects = 0);

    /** Construct a RigidShapedObjectVector
        \param [in] state The simulation state
        \param [in] name Name of the pv
        \param [in] mass Mass of one frozen particle
        \param [in] objSize Number of particles per object
        \param [in] shape The shape that represents the surface of the object
        \param [in] mesh The mesh that represents the surface, should not used in the simulation.
        \param [in] nObjects Number of objects

        \rst
        .. note::
            The mesh is used only for visualization purpose
        \endrst
    */
    RigidShapedObjectVector(const MirState *state, const std::string& name, real mass, int objSize,
                            Shape shape, std::shared_ptr<Mesh> mesh, int nObjects = 0);
    ~RigidShapedObjectVector();

    /// get the handler that represent the shape of the objects
    const Shape& getShape() const {return shape_;}

private:
    Shape shape_;
};

} // namespace mirheo
