// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <vector>
#include <string>

namespace mirheo
{

/** Special channel names used in data managers
 */
namespace channel_names
{

// per entity fields (particles or objects)
extern const std::string globalIds; ///< unique indices

// per particle fields
extern const std::string positions;       ///< coordinates
extern const std::string velocities;      ///< velociries
extern const std::string forces;          ///< forces
extern const std::string stresses;        ///< stresses
extern const std::string densities;       ///< number densities (computed from pairwise density kernels)
extern const std::string oldPositions;    ///< positions at previous time step
extern const std::string polChainVectors; ///< polymeric chain end-to-end vector (see extended DPD interactions)
extern const std::string derChainVectors; ///< time derivative of polymeric chain end-to-end vector

// per object fields
extern const std::string motions;     ///< rigid object states
extern const std::string oldMotions;  ///< rigid object states at previous time step
extern const std::string comExtents;  ///< center of mass and bounding box
extern const std::string areaVolumes; ///< area and volume of membranes

extern const std::string membraneTypeId; ///< Integers to differentiate between groups of membranes

// per object, specific to Juelicher bending + ADE
extern const std::string areas;          ///< area of membranes
extern const std::string meanCurvatures; ///< mean curvature of each membrane
extern const std::string lenThetaTot;    ///< helper quantity to compute the ADE forces

// per bisegment data
extern const std::string polyStates; ///< polymorphic states of each bisegment
extern const std::string energies;   ///< elastic energies of each bisegment
extern const std::string rodKappa;   ///< rod curvature at each bisegment
extern const std::string rodTau_l;   ///< rod torsion at each bisegment

extern const std::vector<std::string> reservedParticleFields;  ///< List of names that should not be used for other particle quantities
extern const std::vector<std::string> reservedObjectFields;    ///< List of names that should not be used for other object quantities
extern const std::vector<std::string> reservedBisegmentFields; ///< List of names that should not be used for other bisegment quantities

/** \param name The name of the channel
    \param reservedNames List of fields that are reserved
    \return \c true if \p name is in \p reservedNames
 */
bool isReserved(const std::string& name, const std::vector<std::string>& reservedNames);

/** \brief dies if \p name is in \p reservedNames
    \param name The name of the channel
    \param reservedNames List of fields that are reserved
 */
void failIfReserved(const std::string& name, const std::vector<std::string>& reservedNames);

/// channel names used in the xdmf files
namespace XDMF
{
extern const std::string position; ///< coordinates
extern const std::string velocity; ///< velocities
extern const std::string ids;      ///< unique ids

/// channels used to pack/unpack rigid motions
namespace motions
{
extern const std::string quaternion; ///< orientation
extern const std::string velocity;   ///< velocity of center of mass
extern const std::string omega;      ///< angular velocity
extern const std::string force;      ///< force acting on the rigid
extern const std::string torque;     ///< torque acting on the rigid
} // namespace motions
} // namespace XDMF
} // namespace channel_names


/// The incrementing method of checkpoint index
enum class CheckpointIdAdvanceMode
{
    PingPong,   ///< oscillate between 0 and 1; this allows to dump a maximum of ywo files per simulation (the last two ones)
    Incremental ///< 0,1,2,... Save all checkpoint files (more memory requirements, but safer)
};


/// Stores the information required to dump checkpoint data
struct CheckpointInfo
{
    /// Constructor
    CheckpointInfo(int every = 0, const std::string& folder = "restart/",
                   CheckpointIdAdvanceMode mode = CheckpointIdAdvanceMode::PingPong);

    /// \return \c true if there will be at least one dump
    bool needDump() const;

    int every; ///< The checkpoint data will be dumped every this many time steps
    std::string folder; ///< target directory (for checkpoints)
    CheckpointIdAdvanceMode mode; ///< The mehod to increment the checkpoint index
};


constexpr int stoppingTag = 424242; ///< tag to notify the postprocess ranks to end the simulation
constexpr int stoppingMsg = -1;     ///< stopping value

constexpr int checkpointTag = 434343; ///< tag to notify the postprocess ranks to perform checkpoint

} // namespace mirheo
