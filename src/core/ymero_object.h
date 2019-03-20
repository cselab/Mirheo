#pragma once

#include "ymero_state.h"

#include <mpi.h>
#include <string>

enum class CheckpointIdAdvanceMode
{
    PingPong,
    Incremental
};

/**
 * Base class for all the objects of YMeRo
 * Only stores name and provides interface for
 * checkpoint / restart mechanism
 */
class YmrObject
{
public:
    YmrObject(std::string name);
    virtual ~YmrObject();

    virtual void checkpoint(MPI_Comm comm, std::string path);  /// Save handler state
    virtual void restart   (MPI_Comm comm, std::string path);  /// Restore handler state

    void advanceCheckpointId(CheckpointIdAdvanceMode mode);
    
public:
    const std::string name;

protected:    
    int checkpointId {0};
};

/**
 * Base class for the objects of YMeRo simulation task
 * may additionally store global quantities in the future
 */
class YmrSimulationObject : public YmrObject
{
public:
    YmrSimulationObject(const YmrState *state, std::string name);
    ~YmrSimulationObject();

public:
    
    const YmrState *state;
};
