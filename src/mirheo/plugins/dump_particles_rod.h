#pragma once

#include "dump_particles.h"

namespace mirheo
{

class RodVector;

class ParticleWithRodQuantitiesSenderPlugin : public ParticleSenderPlugin
{
public:
    
    ParticleWithRodQuantitiesSenderPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery,
                                          const std::vector<std::string>& channelNames);
    
    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;
    
protected:

    RodVector *rv;
    std::map<std::string, DeviceBuffer<real>> channelRodData;
};

} // namespace mirheo
