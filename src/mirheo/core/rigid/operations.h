#pragma once

#include <mirheo/core/pvs/views/rov.h>
#include <mirheo/core/containers.h>

namespace mirheo
{

/// a set of operations applied to \c ROVview objects
namespace rigid_operations
{
/// controls to what quantities to apply the
enum class ApplyTo { PositionsOnly, PositionsAndVelocities };

/// Reduce the forces contained in the particles to the force and torque variable of the RigidMotion objects
void collectRigidForces(const ROVview& view, cudaStream_t stream);

/** Set the positions (and optionally velocities, according to the rigid motions
    \param view The view that contains the input RigidMotion and output particles
    \param initialPositions The positions of the particles in the frame of reference of the object
    \param action Apply the rigid motion to positions or positions and velocities
    \param stream execution stream
    \note The size of \p initialPositions must be the same as the object sizes described by \p view
 */
void applyRigidMotion(const ROVview& view, const PinnedBuffer<real4>& initialPositions,
                      ApplyTo action, cudaStream_t stream);

/// set the force and torques of the RigidMotion objects to zero
void clearRigidForcesFromMotions(const ROVview& view, cudaStream_t stream);

} // namespace rigid_operations

} // namespace mirheo
