#pragma once

#include "domain.h"
#include "utils/common.h"

#include <memory>
#include <mpi.h>
#include <string>

namespace mirheo
{

/**
 * Global quantities accessible by all simulation objects in Mirheo
 */
class MirState
{
public:
    using TimeType = double;
    using StepType = long long;
    
    MirState(DomainInfo domain, real dt);
    MirState(const MirState&);
    MirState& operator=(MirState other);

    virtual ~MirState();

    void swap(MirState& other);

    void reinitTime();

    void checkpoint(MPI_Comm comm, std::string path);  /// Save state to file
    void restart   (MPI_Comm comm, std::string path);  /// Restore state from file

public:
    DomainInfo domain;

    real dt;
    TimeType currentTime;
    StepType currentStep;
};

template <>
struct ConfigDumper<MirState> {
    static Config dump(const MirState& state);
};

} // namespace mirheo
