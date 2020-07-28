// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <vector>

namespace mirheo
{

class ParticleVector;
class RigidObjectVector;

/** Check the validity of all ParticleVector in the simulation:
    - Check that the positions are within reasonable bounds.
    - Check that the velocities are within reasonable bounds.
    - Check that forces do not contain NaN or Inf values.

    If either of the above is not satisfied, the plugin will make the code die with an informative error.
 */
class ParticleCheckerPlugin : public SimulationPlugin
{
public:
    /** Create a ParticleCheckerPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] checkEvery Will check the states of particles every this number of steps.
     */
    ParticleCheckerPlugin(const MirState *state, std::string name, int checkEvery);
    ~ParticleCheckerPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeIntegration(cudaStream_t stream) override;
    void afterIntegration (cudaStream_t stream) override;

    bool needPostproc() override { return false; }

    /// Encode error type if there is any.
    enum class Info {Ok, Out, Nan};

    /// Helper to encode problematic particles.
    struct __align__(16) Status
    {
        int id; ///< The Index of the potential problematic particle.
        Info info; ///< What is problematic.
    };

private:
    void _dieIfBadStatus(cudaStream_t stream, const std::string& identifier);

private:
    int checkEvery_; ///< Particles will be checked every this amount of time steps

    static constexpr int maxNumReports = 256;  ///< maximum number of failed particles info

    /// Contains info related to a given ParticleVector that needs to be checked
    template <class PVType>
    struct CheckData
    {
        PVType *pv;        ///< the ParticleVector to check
        int *numFailedDev; ///< number of failed particles, ptr to device
        int *numFailedHst; ///< number of failed particles, ptr to host
        /// failed particles information
        PinnedBuffer<Status> statuses {maxNumReports};
    };

    using PVCheckData = CheckData<ParticleVector>;
    using ROVCheckData = CheckData<RigidObjectVector>;

    std::vector<PVCheckData> pvCheckData_;
    std::vector<ROVCheckData> rovCheckData_;

    PinnedBuffer<int> numFailed_;
};

} // namespace mirheo
