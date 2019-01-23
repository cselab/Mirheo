#pragma once

#include <memory>
#include <mpi.h>
#include <string>

#include "domain.h"

using TimeType = double;

/**
 * Global quantities accessible by all simulation objects in YMeRo
 */
class YmrState
{
public:
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
    int currentStep;
};

