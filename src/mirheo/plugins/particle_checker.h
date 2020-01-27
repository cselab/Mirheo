#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <vector>

namespace mirheo
{

class ParticleVector;

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
    enum {GoodTag, BadTag};
    
    struct __align__(16) ParticleStatus
    {
        int tag, id;
        Info info;
    };

private:
    void dieIfBadStatus(cudaStream_t stream, const std::string& identifier);
    
private:
    int checkEvery;
    
    PinnedBuffer<ParticleStatus> statuses;
    std::vector<ParticleVector*> pvs;
    std::vector<int> rovStatusIds;

    static constexpr int NotRov = -1;
};

} // namespace mirheo
