#pragma once

#include <string>
#include <mpi.h>

class Simulation;

/**
 * Base class for all the objects of uDeviceX
 * Only stores name and provides interface for
 * checkpoint / restart mechanism
 */
class UdxObject
{
public:
    UdxObject(std::string name) : name(name) {};
    const std::string name;

    virtual void checkpoint(MPI_Comm comm, std::string path) {}  /// Save handler state
    virtual void restart   (MPI_Comm comm, std::string path) {}  /// Restore handler state
    
    virtual ~UdxObject() = default;
};

/**
 * Base class for the objects of uDeviceX simulation task
 * Additionally stores pointer to the managing Simulation
 * Since the objects may be used within different Simulations,
 * need to be able to change the pointer accordingly.
 */
class UdxSimulationObject : public UdxObject
{
public:
    UdxSimulationObject(std::string name) : UdxObject(name) {};
    
    void setSimulation(Simulation* simulation) { this->simulation = simulation; }
    
protected:
    Simulation* simulation;
};
