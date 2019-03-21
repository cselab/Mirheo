#include "ymero_state.h"

#include <core/logger.h>
#include <core/utils/restart_helpers.h>

static const std::string fname = "state.ymero";

YmrState::YmrState(DomainInfo domain, float dt, CheckpointIdAdvanceMode checkpointMode) :
    domain(domain),
    dt(dt),
    currentTime(0),
    currentStep(0),
    checkpointMode(checkpointMode)
{}

YmrState::YmrState(const YmrState&) = default;

YmrState& YmrState::operator=(YmrState other)
{
    swap(other);
    return *this;
}

YmrState::~YmrState() = default;

void YmrState::swap(YmrState& other)
{
    std::swap(domain,      other.domain);
    std::swap(dt,          other.dt);
    std::swap(currentTime, other.currentTime);
    std::swap(currentStep, other.currentStep);
}

static bool isMasterRank(MPI_Comm comm)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    return rank == 0;
}

void YmrState::reinitTime()
{
    currentTime = 0;
    currentStep = 0;
}

void YmrState::checkpoint(MPI_Comm comm, std::string folder)
{
    if (!isMasterRank(comm))
        return;
    
    float3 gsz, gst, lsz;
    gsz = domain.globalSize;
    gst = domain.globalStart;
    lsz = domain.localSize;

    TextIO::write(folder + fname,
                  gsz.x, gsz.y, gsz.z,
                  gst.x, gst.y, gst.z,
                  lsz.x, lsz.y, lsz.z,
                  dt, currentTime, currentStep);
}

void YmrState::restart(MPI_Comm comm, std::string folder)
{
    if (!isMasterRank(comm))
        return;    
    
    float3 gsz, gst, lsz;
    TextIO::read(folder + fname,
                 gsz.x, gsz.y, gsz.z,
                 gst.x, gst.y, gst.z,
                 lsz.x, lsz.y, lsz.z,
                 dt, currentTime, currentStep);

    domain.globalSize  = gsz;
    domain.globalStart = gst;
    domain.localSize   = lsz;
}
