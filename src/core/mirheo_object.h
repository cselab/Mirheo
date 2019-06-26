#pragma once

#include "mirheo_state.h"

#include <core/utils/common.h>

#include <mpi.h>
#include <string>

/**
 * Base class for all the objects of Mirheo
 * Only stores name and provides interface for
 * checkpoint / restart mechanism
 */
class MirObject
{
public:
    MirObject(std::string name);
    virtual ~MirObject();

    virtual void checkpoint(MPI_Comm comm, std::string path, int checkPointId);  /// Save handler state
    virtual void restart   (MPI_Comm comm, std::string path);  /// Restore handler state

    std::string createCheckpointName(std::string path, std::string identifier, std::string extension) const;
    std::string createCheckpointNameWithId(std::string path, std::string identifier, std::string extension, int checkpointId) const;
    void createCheckpointSymlink(MPI_Comm comm, std::string path, std::string identifier, std::string extension, int checkpointId) const;

public:
    const std::string name;
};

/**
 * Base class for the objects of Mirheo simulation task
 * may additionally store global quantities in the future
 */
class MirSimulationObject : public MirObject
{
public:
    MirSimulationObject(const MirState *state, std::string name);
    ~MirSimulationObject();

public:
    const MirState *state;
};
