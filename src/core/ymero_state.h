#pragma once

#include <memory>
#include <mpi.h>
#include <string>

#include "domain.h"

/**
 * Global quantities by all simulation object in YMeRo
 */
class YmrState
{
public:
    YmrState(DomainInfo domain, float dt);
    ~YmrState();

    void checkpoint(MPI_Comm comm, std::string path);  /// Save state to file
    void restart   (MPI_Comm comm, std::string path);  /// Restore state from file

public:
    DomainInfo domain;

    float dt;
    double currentTime{0.0};
    int currentStep{0};
};

