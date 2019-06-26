#pragma once

#include "domain.h"
#include "utils/common.h"

#include <memory>
#include <mpi.h>
#include <string>

/**
 * Global quantities accessible by all simulation objects in YMeRo
 */
class YmrState
{
public:
    using TimeType = double;
    using StepType = long long;
    
    YmrState(DomainInfo domain, float dt);
    YmrState(const YmrState&);
    YmrState& operator=(YmrState other);

    virtual ~YmrState();

    void swap(YmrState& other);
    
    void reinitTime();
    
    void checkpoint(MPI_Comm comm, std::string path);  /// Save state to file
    void restart   (MPI_Comm comm, std::string path);  /// Restore state from file

public:
    DomainInfo domain;

    float dt;
    TimeType currentTime;
    StepType currentStep;
};

