// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "pin_object.h"
#include "utils/simple_serializer.h"

#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/rov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/mpi_types.h>
#include <mirheo/core/utils/path.h>
#include <mirheo/core/utils/quaternion.h>

namespace mirheo
{

namespace pin_object_kernels
{

__global__ void restrictVelocities(OVview view, real3 targetVelocity, real4 *totForces)
{
    int objId = blockIdx.x;

    __shared__ real3 objTotForce, objVelocity;
    objTotForce = make_real3(0.0_r);
    objVelocity = make_real3(0.0_r);
    __syncthreads();

    // Find total force acting on the object and its velocity

    real3 myf = make_real3(0), myv = make_real3(0);
    for (int pid = threadIdx.x; pid < view.objSize; pid += blockDim.x)
    {
        myf += Real3_int(view.forces[pid + objId*view.objSize]).v;
        myv += Real3_int(view.readVelocity(pid + objId*view.objSize)).v;
    }

    myf = warpReduce(myf, [] (real a, real b) { return a+b; });
    myv = warpReduce(myv, [] (real a, real b) { return a+b; });

    if (laneId() == 0)
    {
        atomicAdd(&objTotForce, myf);
        atomicAdd(&objVelocity, myv / view.objSize);  // Average, not simply sum
    }

    __syncthreads();

    // Now only leave the components we need and save the force

    if (threadIdx.x == 0)
    {
        // This is the velocity correction
        objVelocity = targetVelocity - objVelocity;

        if (targetVelocity.x == PinObjectPlugin::Unrestricted) { objTotForce.x = 0; objVelocity.x = 0; }
        if (targetVelocity.y == PinObjectPlugin::Unrestricted) { objTotForce.y = 0; objVelocity.y = 0; }
        if (targetVelocity.z == PinObjectPlugin::Unrestricted) { objTotForce.z = 0; objVelocity.z = 0; }

        totForces[view.ids[objId]] += Real3_int(objTotForce, 0).toReal4();
        objTotForce /= view.objSize;
    }

    __syncthreads();


    // Finally change the original forces and velocities
    // Velocities should be preserved anyways, only changed in the very
    // beginning of the simulation

    for (int pid = threadIdx.x; pid < view.objSize; pid += blockDim.x)
    {
        view.forces    [pid + objId*view.objSize] -= Real3_int(objTotForce, 0).toReal4();
        view.velocities[pid + objId*view.objSize] += Real3_int(objVelocity, 0).toReal4();
    }
}

__global__ void restrictRigidMotion(ROVviewWithOldMotion view, real3 targetVelocity, real3 targetOmega, real dt, real4 *totForces, real4 *totTorques)
{
    int objId = blockIdx.x * blockDim.x + threadIdx.x;
    if (objId >= view.nObjects) return;

    auto motion     = view.motions    [objId];
    auto old_motion = view.old_motions[objId];

    int globObjId = view.ids[objId];

#define VELOCITY_PER_DIM(dim)                                    \
    if (targetVelocity.dim != PinObjectPlugin::Unrestricted)     \
    {                                                            \
        totForces[globObjId].dim += old_motion.force.dim;        \
        motion.r.dim = old_motion.r.dim + targetVelocity.dim*dt; \
        motion.vel.dim = targetVelocity.dim;                     \
    }

    VELOCITY_PER_DIM(x);
    VELOCITY_PER_DIM(y);
    VELOCITY_PER_DIM(z);

#undef VELOCITY_PER_DIM

    // First filter out the invalid values
    auto adjustedTargetOmega = old_motion.omega;
    if (targetOmega.x != PinObjectPlugin::Unrestricted) adjustedTargetOmega.x = targetOmega.x;
    if (targetOmega.y != PinObjectPlugin::Unrestricted) adjustedTargetOmega.y = targetOmega.y;
    if (targetOmega.z != PinObjectPlugin::Unrestricted) adjustedTargetOmega.z = targetOmega.z;

    // Next compute the corrected dq_dt and revert if necessary
    auto dq_dt = old_motion.q.timeDerivative(adjustedTargetOmega);
#define OMEGA_PER_DIM(dim)                                   \
    if (targetOmega.dim != PinObjectPlugin::Unrestricted)    \
    {                                                        \
        totTorques[globObjId].dim += old_motion.torque.dim;  \
        motion.q.dim = old_motion.q.dim + dq_dt.dim*dt;      \
        motion.omega.dim = targetOmega.dim;                  \
    }

    OMEGA_PER_DIM(x);
    OMEGA_PER_DIM(y);
    OMEGA_PER_DIM(z);

#undef OMEGA_PER_DIM

    motion.q.normalize();
    view.motions[objId] = motion;
}

} // namespace pin_object_kernels::

PinObjectPlugin::PinObjectPlugin(const MirState *state, std::string name, std::string ovName, real3 translation, real3 rotation, int reportEvery) :
    SimulationPlugin(state, name),
    ovName_(ovName),
    translation_(translation),
    rotation_(rotation),
    reportEvery_(reportEvery)
{}

void PinObjectPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    ov_ = simulation->getOVbyNameOrDie(ovName_);

    const int myNObj = ov_->local()->getNumObjects();
    int totObjs {0};
    MPI_Check( MPI_Allreduce(&myNObj, &totObjs, 1, MPI_INT, MPI_SUM, comm) );

    forces_.resize_anew(totObjs);
    forces_.clear(defaultStream);

    // Also check torques if object is rigid and if we need to restrict rotation
    rov_ = dynamic_cast<RigidObjectVector*>(ov_);
    if (rov_ != nullptr && (rotation_.x != Unrestricted || rotation_.y != Unrestricted || rotation_.z != Unrestricted))
    {
        torques_.resize_anew(totObjs);
        torques_.clear(defaultStream);
    }

    info("Plugin '%s' is setup for OV '%s' and will impose the following velocity: [%s %s %s]; and following rotation: [%s %s %s]",
         getCName(), ovName_.c_str(),
         translation_.x == Unrestricted ? "?" : std::to_string(translation_.x).c_str(),
         translation_.y == Unrestricted ? "?" : std::to_string(translation_.y).c_str(),
         translation_.z == Unrestricted ? "?" : std::to_string(translation_.z).c_str(),
         rotation_.x == Unrestricted ? "?" : std::to_string(rotation_.x).c_str(),
         rotation_.y == Unrestricted ? "?" : std::to_string(rotation_.y).c_str(),
         rotation_.z == Unrestricted ? "?" : std::to_string(rotation_.z).c_str() );
}


void PinObjectPlugin::handshake()
{
    const bool isRov = (rov_ != nullptr);
    SimpleSerializer::serialize(sendBuffer_, ovName_, isRov);
    _send(sendBuffer_);
}

void PinObjectPlugin::beforeIntegration(cudaStream_t stream)
{
    // If the object is not rigid, modify the forces
    if (rov_ == nullptr)
    {
        debug("Restricting motion of OV '%s' as per plugin '%s'", ovName_.c_str(), getCName());

        const int nthreads = 128;
        OVview view(ov_, ov_->local());
        SAFE_KERNEL_LAUNCH(
                pin_object_kernels::restrictVelocities,
                view.nObjects, nthreads, 0, stream,
                view, translation_, forces_.devPtr() );
    }
}

void PinObjectPlugin::afterIntegration(cudaStream_t stream)
{
    // If the object IS rigid, modify forces and torques
    if (rov_ != nullptr)
    {
        debug("Restricting rigid motion of OV '%s' as per plugin '%s'", ovName_.c_str(), getCName());

        const int nthreads = 32;
        ROVviewWithOldMotion view(rov_, rov_->local());
        SAFE_KERNEL_LAUNCH(
                pin_object_kernels::restrictRigidMotion,
                getNblocks(view.nObjects, nthreads), nthreads, 0, stream,
                view, translation_, rotation_, getState()->getDt(),
                forces_.devPtr(), torques_.devPtr() );
    }
}

void PinObjectPlugin::serializeAndSend(cudaStream_t stream)
{
    count_++;
    if (count_ % reportEvery_ != 0) return;

    forces_.downloadFromDevice(stream);
    if (rov_ != nullptr)
        torques_.downloadFromDevice(stream);

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, getState()->currentTime, reportEvery_, forces_, torques_);
    _send(sendBuffer_);

    forces_.clearDevice(stream);
    torques_.clearDevice(stream);
}


ReportPinObjectPlugin::ReportPinObjectPlugin(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path))
{}

void ReportPinObjectPlugin::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void ReportPinObjectPlugin::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string ovName;
    bool isRov;
    SimpleSerializer::deserialize(data_, ovName, isRov);

    if (activated_ && rank_ == 0 && fout_.get() == nullptr)
    {
        const std::string fname = joinPaths(path_, setExtensionOrDie(ovName, "csv"));
        auto status = fout_.open(fname, "w" );
        if (status != FileWrapper::Status::Success)
            die("could not open file '%s'", fname.c_str());

        // print header
        fprintf(fout_.get(), "objId,time,fx,fy,fz");

        if (isRov)
            fprintf(fout_.get(), ",Tx,Ty,Tz");

        fprintf(fout_.get(), "\n");
    }
}

void ReportPinObjectPlugin::deserialize()
{
    std::vector<real4> forces, torques;
    MirState::TimeType currentTime;
    int nsamples;

    SimpleSerializer::deserialize(data_, currentTime, nsamples, forces, torques);

    MPI_Check( MPI_Reduce( (rank_ == 0 ? MPI_IN_PLACE : forces.data()),  forces.data(),  forces.size()*4,  getMPIFloatType<real>(), MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce( (rank_ == 0 ? MPI_IN_PLACE : torques.data()), torques.data(), torques.size()*4, getMPIFloatType<real>(), MPI_SUM, 0, comm_) );

    if (activated_ && rank_ == 0)
    {
        for (size_t i = 0; i < forces.size(); ++i)
        {
            forces[i] /= nsamples;
            fprintf(fout_.get(), "%lu,%f,%f,%f,%f",
                    i, currentTime, forces[i].x, forces[i].y, forces[i].z);

            if (i < torques.size())
            {
                torques[i] /= nsamples;
                fprintf(fout_.get(), ",%f,%f,%f", torques[i].x, torques[i].y, torques[i].z);
            }

            fprintf(fout_.get(), "\n");
        }

        fflush(fout_.get());
    }
}

} // namespace mirheo
