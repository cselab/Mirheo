// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "mirheo_state.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/restart_helpers.h>

namespace mirheo
{

static const std::string fname = "state.mirheo";

MirState::MirState(DomainInfo domain_, real dt) :
    domain(domain_),
    currentTime(0),
    currentStep(0),
    dt_(dt)
{}

static bool isMasterRank(MPI_Comm comm)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    return rank == 0;
}

void MirState::checkpoint(MPI_Comm comm, std::string folder)
{
    if (!isMasterRank(comm))
        return;

    real3 gsz, gst, lsz;
    gsz = domain.globalSize;
    gst = domain.globalStart;
    lsz = domain.localSize;

    text_IO::write(folder + fname,
                  gsz.x, gsz.y, gsz.z,
                  gst.x, gst.y, gst.z,
                  lsz.x, lsz.y, lsz.z,
                  dt_, currentTime, currentStep);
}

void MirState::restart(MPI_Comm comm, std::string folder)
{
    if (!isMasterRank(comm))
        return;

    real3 gsz, gst, lsz;
    auto filename = folder + fname;
    auto good = text_IO::read(filename,
                             gsz.x, gsz.y, gsz.z,
                             gst.x, gst.y, gst.z,
                             lsz.x, lsz.y, lsz.z,
                             dt_, currentTime, currentStep);

    if (!good) die("failed to read '%s'\n", filename.c_str());

    domain.globalSize  = gsz;
    domain.globalStart = gst;
    domain.localSize   = lsz;
}

void MirState::_dieInvalidDt [[noreturn]]() const {
    die("Time step dt not available. Using dt is valid only during Mirheo::run().");
}

} // namespace mirheo
