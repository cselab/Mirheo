#include "force_saver.h"

#include <core/simulation.h>
#include <core/pvs/particle_vector.h>

const std::string ForceSaverPlugin::fieldName = "forces";

ForceSaverPlugin::ForceSaverPlugin(std::string name, std::string pvName) :
    SimulationPlugin(name), pvName(pvName), pv(nullptr)
{}

void ForceSaverPlugin::beforeIntegration(cudaStream_t stream)
{
    pv->local()->extraPerParticle.getData<Force>(fieldName);
}
    
bool ForceSaverPlugin::needPostproc()
{
    return false;
}

void ForceSaverPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    pv->requireDataPerParticle<Force>(fieldName, false);
}

    
