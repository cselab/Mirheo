#pragma once

#include <string>
#include <mpi.h>

class Simulation;

/**
 * Base class for all the objects of YMeRo
 * Only stores name and provides interface for
 * checkpoint / restart mechanism
 */
class YmrObject
{
public:
    YmrObject(std::string name) : name(name) {};
    const std::string name;

    virtual void checkpoint(MPI_Comm comm, std::string path) {}  /// Save handler state
    virtual void restart   (MPI_Comm comm, std::string path) {}  /// Restore handler state
    
    virtual ~YmrObject() = default;
};

/**
 * Base class for the objects of YMeRo simulation task
 * Additionally stores pointer to the managing Simulation
 * Since the objects may be used within different Simulations,
 * need to be able to change the pointer accordingly.
 */
class YmrSimulationObject : public YmrObject
{
public:
    YmrSimulationObject(std::string name) : YmrObject(name) {};
    
    void setSimulation(Simulation* simulation) { this->simulation = simulation; }
    
protected:
    Simulation* simulation;
};
