// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/path.h>

#include <functional>
#include <string>

namespace mirheo
{

class RigidObjectVector;

/** Apply a magnetic torque on given a RigidObjectVector.
 */
class ExternalMagneticTorquePlugin : public SimulationPlugin
{
public:

    /// Time varying uniform field.
    using UniformMagneticFunc = std::function<real3(real)>;

    /** Create a ExternalMagneticTorquePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] rovName The name of the RigidObjectVector to apply the torque to.
        \param [in] moment The constant magnetic moment of one object, in its frame of reference.
        \param [in] magneticFunction The external uniform magnetic field which possibly varies in time.
     */
    ExternalMagneticTorquePlugin(const MirState *state, std::string name,
                                 std::string rovName, real3 moment, UniformMagneticFunc magneticFunction);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string rovName_;
    RigidObjectVector *rov_;
    real3 moment_;
    UniformMagneticFunc magneticFunction_;
};

} // namespace mirheo
