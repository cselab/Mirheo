#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <vector>

namespace mirheo
{

class ParticleVector;
class RigidObjectVector;

/** \brief This \c Plugin has only a debug purpose.
    It is designed to check the validity of the ParticleVector states:
    - check the positions are within reasonable bounds
    - check the velocities are within reasonable bounds
    - check for nan or inf forces
 */
class ParticleCheckerPlugin : public SimulationPlugin
{
public:
    ParticleCheckerPlugin(const MirState *state, std::string name, int checkEvery);
    ~ParticleCheckerPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    
    void beforeIntegration(cudaStream_t stream) override;
    void afterIntegration (cudaStream_t stream) override;

    bool needPostproc() override { return false; }

    enum class Info {Ok, Out, Nan};
    
    struct __align__(16) Status
    {
        int id;
        Info info;
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
