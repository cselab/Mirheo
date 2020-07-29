// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/utils/path.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace mirheo
{

class ObjectVector;
class RigidObjectVector;

/** Add constraints on objects of an ObjectVector.
    This modifies the velocities and forces on the objects in order to satisfy the given constraints on
    linear and angular velocities.

    This plugin also collects statistics on the required forces and torques used to maintain the constraints.
    This may be useful to e.g. measure the drag around objects.
    See ReportPinObjectPlugin.
 */
class PinObjectPlugin : public SimulationPlugin
{
public:
    /// Special value reserved to represent unrestricted components.
    constexpr static real Unrestricted = std::numeric_limits<real>::infinity();

    /** Create a PinObjectPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] ovName The name of the ObjectVector that will be subjected of the constraints.
        \param [in] translation The target linear velocity. Components set to Unrestricted will not be constrained.
        \param [in] rotation The target angular velocity. Components set to Unrestricted will not be constrained.
        \param [in] reportEvery Send forces and torques stats to the postprocess side every this number of time steps.
     */
    PinObjectPlugin(const MirState *state, std::string name, std::string ovName, real3 translation, real3 rotation, int reportEvery);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeIntegration(cudaStream_t stream) override;
    void afterIntegration (cudaStream_t stream) override;
    void serializeAndSend (cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string ovName_;
    ObjectVector *ov_{nullptr};
    RigidObjectVector *rov_{nullptr};

    real3 translation_, rotation_;

    int reportEvery_;
    int count_{0};

    PinnedBuffer<real4> forces_, torques_;
    std::vector<char> sendBuffer_;
};

/** Postprocess side of PinObjectPlugin.
    Receives and umps the forces and torques required to keep the constraints satsfied.
*/
class ReportPinObjectPlugin : public PostprocessPlugin
{
public:
    /** Create a PinObjectPlugin object.
        \param [in] name The name of the plugin.
        \param [in] path Path to the csv file that will contain the statistics.
     */
    ReportPinObjectPlugin(std::string name, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    bool activated_;
    std::string path_;

    FileWrapper fout_;
};

} // namespace mirheo
