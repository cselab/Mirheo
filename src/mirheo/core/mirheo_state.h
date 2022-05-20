// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "domain.h"
#include "utils/common.h"

#include <mpi.h>
#include <string>

namespace mirheo
{

/** \brief Global quantities accessible by all simulation objects in Mirheo
 */
class MirState
{
public:
    static constexpr real InvalidDt = -1; ///< Special value used to initialize invalid dt.
    using TimeType = double; ///< type used to store time information
    using StepType = long long; ///< type to store time step information

    /** \brief Construct a MirState object
        \param [in] domain The DomainInfo of the simulation
        \param [in] dt Simulation time step
    */
    MirState(DomainInfo domain, real dt = (real)InvalidDt);


    /** Save internal state to file
        \param [in] comm MPI comm of the simulation
        \param [in] path The directory in which to save the file
     */
    void checkpoint(MPI_Comm comm, std::string path);

    /** Load internal state from file
        \param [in] comm MPI comm of the simulation
        \param [in] path The directory from which to load the file
     */
    void restart(MPI_Comm comm, std::string path);

    /** Get the current time step dt. Accessible only during Mirheo::run. */
    real getDt() const {
        if (dt_ < 0)
            _dieInvalidDt();
        return dt_;
    }

    /** Set the time step dt.
        \param [in] dt time step duration
     */
    void setDt(real dt) noexcept {
        dt_ = dt;
    }

public:
    DomainInfo domain; ///< Global DomainInfo

    TimeType currentTime; ///< Current simulation time
    StepType currentStep; ///< Current simulation step

private:
    void _dieInvalidDt [[noreturn]]() const; // To avoid including logger here.
    real dt_; ///< time step
};

} // namespace mirheo
