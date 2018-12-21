#pragma once

#include <string>
#include <mpi.h>

#include "ymero_state.h"

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
 * may additionally store global quantities in the future
 */
class YmrSimulationObject : public YmrObject
{
public:
    YmrSimulationObject(const YmrState *state, std::string name) :
        YmrObject(name), state(state)
    {}

    const YmrState *state;
};
