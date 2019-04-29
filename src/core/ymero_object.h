#pragma once

#include "ymero_state.h"

#include <core/utils/common.h>

#include <mpi.h>
#include <string>

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

    virtual void checkpoint(MPI_Comm comm, std::string path, int checkPointId);  /// Save handler state
    virtual void restart   (MPI_Comm comm, std::string path);  /// Restore handler state

    std::string createCheckpointName(std::string path, std::string identifier, std::string extension) const;
    std::string createCheckpointNameWithId(std::string path, std::string identifier, std::string extension, int checkpointId) const;
    void createCheckpointSymlink(MPI_Comm comm, std::string path, std::string identifier, std::string extension, int checkpointId) const;

public:
    const std::string name;
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
