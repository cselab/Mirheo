#include "particle_checker.h"
#include "utils/time_stamp.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/pvs/views/rov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/types/str.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/strprintf.h>

namespace mirheo
{

namespace ParticleCheckerKernels
{
template<typename R3>
__device__ static inline bool isFinite(R3 v)
{
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

template<typename R3>
__device__ static inline bool withinBounds(R3 v, real3 bounds)
{
    return
        (math::abs(v.x) < bounds.x) &&
        (math::abs(v.y) < bounds.y) &&
        (math::abs(v.z) < bounds.z);
}

template <int maxNumReports>
__device__ static inline void setBadStatus(int pid, ParticleCheckerPlugin::Info info,
                                           int *numFailed, ParticleCheckerPlugin::Status *statuses)
{
    const int failedId = atomicAdd(numFailed, 1);

    if (failedId < maxNumReports)
    {
        statuses[failedId].id   = pid;
        statuses[failedId].info = info;
    }
}

template <int maxNumReports>
__global__ void checkForces(PVview view, int *numFailed, ParticleCheckerPlugin::Status *statuses)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;

    if (pid >= view.size) return;

    const auto force = make_real3(view.forces[pid]);

    if (!isFinite(force))
        setBadStatus<maxNumReports>(pid, ParticleCheckerPlugin::Info::Nan, numFailed, statuses);
}

template <int maxNumReports>
__global__ void checkParticles(PVview view, DomainInfo domain, real dtInv, int *numFailed, ParticleCheckerPlugin::Status *statuses)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;

    if (pid >= view.size) return;

    const auto pos = make_real3(view.readPosition(pid));
    const auto vel = make_real3(view.readVelocity(pid));

    if (!isFinite(pos) || !isFinite(vel))
    {
        setBadStatus<maxNumReports>(pid, ParticleCheckerPlugin::Info::Nan, numFailed, statuses);
        return;
    }

    const real3 boundsPos = 1.5_r * domain.localSize; // particle should not be further than one neighbouring domain
    const real3 boundsVel = dtInv * domain.localSize; // particle should not travel more than one domain size per iteration

    if (!withinBounds(pos, boundsPos) || !withinBounds(vel, boundsVel))
    {
        setBadStatus<maxNumReports>(pid, ParticleCheckerPlugin::Info::Out, numFailed, statuses);
        return;
    }
}

template <int maxNumReports>
__global__ void checkRigidForces(ROVview view, int *numFailed, ParticleCheckerPlugin::Status *statuses)
{
    const int objId = blockIdx.x * blockDim.x + threadIdx.x;

    if (objId >= view.nObjects) return;

    const auto m = view.motions[objId];

    if (!isFinite(m.force) || !isFinite(m.torque))
        setBadStatus<maxNumReports>(objId, ParticleCheckerPlugin::Info::Nan, numFailed, statuses);
}

template <int maxNumReports>
__global__ void checkRigidMotions(ROVview view, DomainInfo domain, real dtInv, int *numFailed, ParticleCheckerPlugin::Status *statuses)
{
    const int objId = blockIdx.x * blockDim.x + threadIdx.x;

    if (objId >= view.nObjects) return;

    const auto m = view.motions[objId];

    if (!isFinite(m.r) || !isFinite(m.vel) || !isFinite(m.omega))
    {
        setBadStatus<maxNumReports>(objId, ParticleCheckerPlugin::Info::Nan, numFailed, statuses);
        return;
    }

    const real3 boundsPos   = 1.5_r * domain.localSize; // objects should not be further than one neighbouring domain
    const real3 boundsVel   = dtInv * domain.localSize; // objects should not travel more than one domain size per iteration
    const real3 boundsOmega = make_real3(dtInv * M_PI); // objects should not rotate more than half a turn per iteration

    if (!withinBounds(m.r, boundsPos) || !withinBounds(m.vel, boundsVel), !withinBounds(m.omega, boundsOmega))
    {
        setBadStatus<maxNumReports>(objId, ParticleCheckerPlugin::Info::Out, numFailed, statuses);
        return;
    }
}

} // namespace ParticleCheckerKernels

constexpr int ParticleCheckerPlugin::maxNumReports;

ParticleCheckerPlugin::ParticleCheckerPlugin(const MirState *state, std::string name, int checkEvery) :
    SimulationPlugin(state, name),
    checkEvery_(checkEvery)
{}

ParticleCheckerPlugin::~ParticleCheckerPlugin() = default;

void ParticleCheckerPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    auto pvs = simulation->getParticleVectors();
    
    numFailed_.resize_anew(pvs.size() * 2); // 2 * pvs.size >= that number of pvs + number of rovs
    numFailed_.clear(defaultStream);
    
    int *numFailedDevPtr = numFailed_.devPtr();
    int *numFailedHstPtr = numFailed_.hostPtr();

    for (auto pv : pvs)
    {
        PVCheckData pvCd;
        pvCd.pv = pv;
        pvCd.numFailedDev = numFailedDevPtr++; 
        pvCd.numFailedHst = numFailedHstPtr++;
        pvCheckData_.push_back(std::move(pvCd));

        if (auto rov = dynamic_cast<RigidObjectVector*>(pv))
        {
            ROVCheckData rovCd;
            rovCd.pv = rov;
            rovCd.numFailedDev = numFailedDevPtr++; 
            rovCd.numFailedHst = numFailedHstPtr++;
            rovCheckData_.push_back(std::move(rovCd));
        }
    }
}

void ParticleCheckerPlugin::beforeIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), checkEvery_)) return;

    constexpr int nthreads = 128;
    
    for (auto& pvCd : pvCheckData_)
    {
        auto pv = pvCd.pv;
        PVview view(pv, pv->local());

        SAFE_KERNEL_LAUNCH(
            ParticleCheckerKernels::checkForces<maxNumReports>,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, pvCd.numFailedDev, pvCd.statuses.devPtr() );
    }

    for (auto& rovCd : rovCheckData_)
    {
        auto rov = rovCd.pv;
        ROVview rovView(rov, rov->local());
        
        SAFE_KERNEL_LAUNCH(
            ParticleCheckerKernels::checkRigidForces<maxNumReports>,
            getNblocks(rovView.nObjects, nthreads), nthreads, 0, stream,
            rovView, rovCd.numFailedDev, rovCd.statuses.devPtr() );
    }

    _dieIfBadStatus(stream, "force");
}

void ParticleCheckerPlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), checkEvery_)) return;

    constexpr int nthreads = 128;

    const real dt     = getState()->dt;
    const real dtInv  = 1.0_r / math::max(1e-6_r, dt);
    const auto domain = getState()->domain;
    
    for (auto& pvCd : pvCheckData_)
    {
        auto pv = pvCd.pv;
        PVview view(pv, pv->local());

        SAFE_KERNEL_LAUNCH(
            ParticleCheckerKernels::checkParticles<maxNumReports>,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, domain, dtInv, pvCd.numFailedDev, pvCd.statuses.devPtr() );
    }

    for (auto& rovCd : rovCheckData_)
    {
        auto rov = rovCd.pv;
        ROVview rovView(rov, rov->local());

        SAFE_KERNEL_LAUNCH(
            ParticleCheckerKernels::checkRigidMotions<maxNumReports>,
            getNblocks(rovView.nObjects, nthreads), nthreads, 0, stream,
            rovView, domain, dtInv, rovCd.numFailedDev, rovCd.statuses.devPtr() );
    }

    _dieIfBadStatus(stream, "particle");
}

static inline void downloadAllFields(cudaStream_t stream, const DataManager& manager)
{
    for (auto entry : manager.getSortedChannels())
    {
        auto desc = entry.second;
        mpark::visit([stream](auto pinnedBuffPtr)
        {
            pinnedBuffPtr->downloadFromDevice(stream, ContainersSynch::Asynch);
        }, desc->varDataPtr);
    }
    CUDA_Check( cudaStreamSynchronize(stream) );
}

static inline std::string listOtherFieldValues(const DataManager& manager, int id)
{
    std::string fieldValues;
    
    for (auto entry : manager.getSortedChannels())
    {
        const auto& name = entry.first;
        const auto desc = entry.second;
            
        if (name == ChannelNames::positions ||
            name == ChannelNames::velocities)
            continue;
            
        mpark::visit([&](auto pinnedBuffPtr)
        {
            const auto val = (*pinnedBuffPtr)[id];
            fieldValues += '\t' + name + " : " + printToStr(val) + '\n';
        }, desc->varDataPtr);
    }
    return fieldValues;    
}

static inline std::string infoToStr(ParticleCheckerPlugin::Info info)
{
    using Info = ParticleCheckerPlugin::Info;
    if (info == Info::Nan) return "not a finite number";
    if (info == Info::Out) return "out of bounds";
    return "no error detected";
}

void ParticleCheckerPlugin::_dieIfBadStatus(cudaStream_t stream, const std::string& identifier)
{
    numFailed_.downloadFromDevice(stream, ContainersSynch::Synch);
    const auto domain = getState()->domain;

    bool failing {false};
    std::string allErrors;

    for (auto& pvCd : pvCheckData_)
    {
        const int numFailed = *pvCd.numFailedHst;
        if (numFailed == 0) continue;

        // from now we know we will fail; download data and print error
        pvCd.statuses.downloadFromDevice(stream, ContainersSynch::Asynch); // async because downloadAllFields will sync that stream

        auto pv = pvCd.pv;
        auto lpv = pv->local();

        downloadAllFields(stream, lpv->dataPerParticle);

        for (int i = 0; i < std::min(numFailed, maxNumReports); ++i)
        {
            const auto s = pvCd.statuses[i];
            const int partId = s.id;
            const auto p = Particle(lpv->positions ()[partId],
                                    lpv->velocities()[partId]);

            const auto infoStr = infoToStr(s.info);

            const real3 lr = p.r;
            const real3 gr = domain.local2global(lr);

            allErrors += strprintf("\n\tBad %s in '%s' with id %ld, local position %g %g %g, global position %g %g %g, velocity %g %g %g : %s\n",
                                   identifier.c_str(),
                                   pv->getCName(), p.getId(),
                                   lr.x, lr.y, lr.z, gr.x, gr.y, gr.z,
                                   p.u.x, p.u.y, p.u.z, infoStr.c_str());

            allErrors += listOtherFieldValues(lpv->dataPerParticle, partId);
        }
        
        failing = true;
    }

    for (auto& rovCd : rovCheckData_)
    {
        const int numFailed = *rovCd.numFailedHst;
        if (numFailed == 0) continue;

        // from now we know we will fail; download data and print error

        rovCd.statuses.downloadFromDevice(stream, ContainersSynch::Asynch); // async because downloadAllFields will sync that stream

        auto rov = rovCd.pv;
        auto lrov = rov->local();

        downloadAllFields(stream, lrov->dataPerObject);

        for (int i = 0; i < std::min(numFailed, maxNumReports); ++i)
        {
            const auto s = rovCd.statuses[i];
            const int rovId = s.id;
            const auto infoStr = infoToStr(s.info);
        
            allErrors += strprintf("\n\tBad %s in rov '%s' : %s\n",
                                   identifier.c_str(), rov->getCName(), infoStr.c_str());

            allErrors += listOtherFieldValues(lrov->dataPerObject, rovId);
        }
        
        failing = true;
    }

    if (failing)
        die("Particle checker has found bad particles: %s", allErrors.c_str());
}

} // namespace mirheo
