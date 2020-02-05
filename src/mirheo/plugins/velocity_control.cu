#include <mirheo/core/utils/restart_helpers.h>

#include "velocity_control.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace VelocityControlKernels
{

inline __device__ bool is_inside(real3 r, real3 low, real3 high)
{
    return
        low.x <= r.x && r.x <= high.x &&
        low.y <= r.y && r.y <= high.y &&
        low.z <= r.z && r.z <= high.z;
}

__global__ void addForce(PVview view, DomainInfo domain, real3 low, real3 high, real3 force)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    auto r = Real3_int(view.readPosition(gid)).v;
    
    const real3 gr = domain.local2global(r);

    if (is_inside(gr, low, high))
        view.forces[gid] += make_real4(force, 0.0_r);
}

__global__ void sumVelocity(PVview view, DomainInfo domain, real3 low, real3 high, real3 *totVel, int *nSamples)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    Particle p;
    
    p.u = make_real3(0.0_r);

    if (gid < view.size) {

        p = view.readParticle(gid);
        const real3 gr = domain.local2global(p.r);

        if (is_inside(gr, low, high))
            atomicAggInc(nSamples);
        else
            p.u = make_real3(0.0_r);
    }

    const real3 u = warpReduce(p.u, [](real a, real b) { return a+b; });
    
    if (laneId() == 0 && dot(u, u) > 1e-8)
        atomicAdd(totVel, u);
}

} // namespace VelocityControlKernels

SimulationVelocityControl::SimulationVelocityControl(const MirState *state, std::string name, std::vector<std::string> pvNames,
                                                     real3 low, real3 high,
                                                     int sampleEvery, int tuneEvery, int dumpEvery,
                                                     real3 targetVel, real Kp, real Ki, real Kd) :
    SimulationPlugin(state, name),
    pvNames_(pvNames),
    low_(low),
    high_(high),
    currentVel_(make_real3(0,0,0)),
    targetVel_(targetVel),
    sampleEvery_(sampleEvery),
    tuneEvery_(tuneEvery),
    dumpEvery_(dumpEvery), 
    force_(make_real3(0, 0, 0)),
    pid_(make_real3(0, 0, 0), Kp, Ki, Kd),
    accumulatedTotVel_({0,0,0})
{}


void SimulationVelocityControl::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    for (auto &pvName : pvNames_)
        pvs_.push_back(simulation->getPVbyNameOrDie(pvName));
}

void SimulationVelocityControl::beforeForces(cudaStream_t stream)
{
    for (auto &pv : pvs_)
    {
        PVview view(pv, pv->local());
        const int nthreads = 128;

        SAFE_KERNEL_LAUNCH
            (VelocityControlKernels::addForce,
             getNblocks(view.size, nthreads), nthreads, 0, stream,
             view, getState()->domain, low_, high_, force_ );
    }
}

void SimulationVelocityControl::_sampleOnePv(ParticleVector *pv, cudaStream_t stream) {
    PVview pvView(pv, pv->local());
    const int nthreads = 128;
 
    SAFE_KERNEL_LAUNCH
        (VelocityControlKernels::sumVelocity,
         getNblocks(pvView.size, nthreads), nthreads, 0, stream,
         pvView, getState()->domain, low_, high_, totVel_.devPtr(), nSamples_.devPtr());
}

void SimulationVelocityControl::afterIntegration(cudaStream_t stream)
{
    if (isTimeEvery(getState(), sampleEvery_))
    {
        debug2("Velocity control %s is sampling now", getCName());

        totVel_.clearDevice(stream);
        for (auto &pv : pvs_)
            _sampleOnePv(pv, stream);
        totVel_.downloadFromDevice(stream);
        accumulatedTotVel_.x += totVel_[0].x;
        accumulatedTotVel_.y += totVel_[0].y;
        accumulatedTotVel_.z += totVel_[0].z;
    }
    
    if (!isTimeEvery(getState(), tuneEvery_)) return;
    
    nSamples_.downloadFromDevice(stream);
    nSamples_.clearDevice(stream);
    
    long nSamplesTot = 0;
    double3 totVelTot = make_double3(0,0,0);

    const long nSamplesLoc = nSamples_[0];
    
    MPI_Check( MPI_Allreduce(&nSamplesLoc,         &nSamplesTot, 1, MPI_LONG,   MPI_SUM, comm) );
    MPI_Check( MPI_Allreduce(&accumulatedTotVel_,  &totVelTot,   3, MPI_DOUBLE, MPI_SUM, comm) );

    currentVel_ = nSamplesTot ? make_real3(totVelTot / nSamplesTot) : make_real3(0._r, 0._r, 0._r);
    force_ = pid_.update(targetVel_ - currentVel_);
    accumulatedTotVel_ = {0,0,0};
}

void SimulationVelocityControl::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, getState()->currentTime, getState()->currentStep, currentVel_, force_);
    send(sendBuffer_);
}

void SimulationVelocityControl::checkpoint(MPI_Comm comm, const std::string& path, int checkpointId)
{
    const auto filename = createCheckpointNameWithId(path, "plugin." + getName(), "txt", checkpointId);

    TextIO::write(filename, pid_);
    
    createCheckpointSymlink(comm, path, "plugin." + getName(), "txt", checkpointId);
}

void SimulationVelocityControl::restart(__UNUSED MPI_Comm comm, const std::string& path)
{
    const auto filename = createCheckpointName(path, "plugin." + getName(), "txt");
    const bool good = TextIO::read(filename, pid_);
    if (!good) die("failed to read '%s'\n", filename.c_str());
}




PostprocessVelocityControl::PostprocessVelocityControl(std::string name, std::string filename) :
    PostprocessPlugin(name)
{
    auto status = fdump_.open(filename, "w");
    if (status != FileWrapper::Status::Success)
        die("Could not open file '%s'", filename.c_str());
    fprintf(fdump_.get(), "# time time_step velocity force\n");
}

void PostprocessVelocityControl::deserialize()
{
    MirState::StepType currentTimeStep;
    MirState::TimeType currentTime;
    real3 vel, force;

    SimpleSerializer::deserialize(data, currentTime, currentTimeStep, vel, force);

    if (rank == 0) {
        fprintf(fdump_.get(),
                "%g %lld "
                "%g %g %g "
                "%g %g %g\n",
                currentTime, currentTimeStep,
                vel.x, vel.y, vel.z,
                force.x, force.y, force.z);
        
        fflush(fdump_.get());
    }
}

} // namespace mirheo
