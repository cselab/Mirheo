#include "pin_object.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/views/rov.h>
#include <core/simulation.h>

#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/rigid_kernels/quaternion.h>

#include "simple_serializer.h"
#include "pin_object.h"

__global__ void restrictVelocities(OVview view, float3 targetVelocity, float4* totForces)
{
    int objId = blockIdx.x;
    
    __shared__ float3 objTotForce, objVelocity;
    objTotForce = make_float3(0);
    objVelocity = make_float3(0);
    __syncthreads();

    // Find total force acting on the object and its velocity

    float3 myf = make_float3(0), myv = make_float3(0);
    for (int pid = threadIdx.x; pid < view.objSize; pid += blockDim.x)
    {
        myf += Float3_int(view.forces[pid + objId*view.objSize]).v;
        myv += Float3_int(view.particles[2*(pid + objId*view.objSize) + 1]).v;
    }

    myf = warpReduce(myf, [] (float a, float b) { return a+b; });
    myv = warpReduce(myv, [] (float a, float b) { return a+b; });

    if (__laneid() == 0)
    {
        atomicAdd(&objTotForce, myf);
        atomicAdd(&objVelocity, myv / view.objSize);  // Average, not simply sum
    }

    __syncthreads();

    // Now only leave the components we need and save the force

    float3 extraForce = make_float3(0);
    if (targetVelocity.x == PinObjectPlugin::Unrestricted) { objTotForce.x = 0; objVelocity.x = 0; }
    if (targetVelocity.y == PinObjectPlugin::Unrestricted) { objTotForce.y = 0; objVelocity.y = 0; }
    if (targetVelocity.z == PinObjectPlugin::Unrestricted) { objTotForce.z = 0; objVelocity.z = 0; }

    if (threadIdx.x == 0)
        totForces[view.ids[objId]] += Float3_int(objTotForce, 0).toFloat4();

    // Finally change the original forces and velocities
    // Velocities should be preserved anyways, only changed in the very
    // beginning of the simulation
    objTotForce /= view.objSize;

    for (int pid = threadIdx.x; pid < view.objSize; pid += blockDim.x)
        view.forces[pid + objId*view.objSize] -= Float3_int(objTotForce, 0).toFloat4();
}

__global__ void restrictRigidMotion(ROVviewWithOldMotion view, float3 targetVelocity, float3 targetOmega, float dt, float4* totForces, float4* totTorques)
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
    
    
    // https://stackoverflow.com/a/22401169/3535276
    // looks like q.x, 0, 0, q.w is responsible for the X axis rotation etc.
    // so to restrict rotation along ie. X, we need to preserve q.x
    // and normalize of course
    
    // First filter out the invalid values
    auto adjustedTargetOmega = old_motion.omega;
    if (targetOmega.x != PinObjectPlugin::Unrestricted) adjustedTargetOmega.x = targetOmega.x;
    if (targetOmega.y != PinObjectPlugin::Unrestricted) adjustedTargetOmega.y = targetOmega.y;
    if (targetOmega.z != PinObjectPlugin::Unrestricted) adjustedTargetOmega.z = targetOmega.z;
    
    // Next compute the corrected dq_dt and revert if necessary
    auto dq_dt = compute_dq_dt(old_motion.q, adjustedTargetOmega);
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
    
    motion.q = normalize(motion.q);
    view.motions[objId] = motion;
}


PinObjectPlugin::PinObjectPlugin(std::string name, std::string ovName, float3 translation, float3 rotation, int reportEvery) :
    SimulationPlugin(name), ovName(ovName),
    translation(translation), rotation(rotation),
    reportEvery(reportEvery)
{    }

void PinObjectPlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(sim, comm, interComm);

    ov = sim->getOVbyNameOrDie(ovName);

    int myNObj = ov->local()->nObjects;
    int totObjs;
    MPI_Check( MPI_Allreduce(&myNObj, &totObjs, 1, MPI_INT, MPI_SUM, comm) );

    forces.resize_anew(totObjs);
    forces.clear(0);

    // Also check torques if object is rigid and if we need to restrict rotation
    rov = dynamic_cast<RigidObjectVector*>(ov);
    if (rov != nullptr && (rotation.x != Unrestricted || rotation.y != Unrestricted || rotation.z != Unrestricted))
    {
        torques.resize_anew(totObjs);
        torques.clear(0);
    }

    debug("Plugin PinObject is setup for OV '%s' and will impose the following velocity: [%s %s %s]; and following rotation: [%s %s %s]",
          ovName.c_str(),

          translation.x == Unrestricted ? "?" : std::to_string(translation.x).c_str(),
          translation.y == Unrestricted ? "?" : std::to_string(translation.y).c_str(),
          translation.z == Unrestricted ? "?" : std::to_string(translation.z).c_str(),

          rotation.x == Unrestricted ? "?" : std::to_string(rotation.x).c_str(),
          rotation.y == Unrestricted ? "?" : std::to_string(rotation.y).c_str(),
          rotation.z == Unrestricted ? "?" : std::to_string(rotation.z).c_str() );
}


void PinObjectPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer, ovName);
    send(sendBuffer);
}

void PinObjectPlugin::beforeIntegration(cudaStream_t stream)
{
    // If the object is not rigid, modify the forces
    if (rov == nullptr)
    {
        debug("Restricting motion of OV '%s' as per plugin '%s'", ovName.c_str(), name.c_str());

        const int nthreads = 128;
        OVview view(ov, ov->local());
        SAFE_KERNEL_LAUNCH(
                restrictVelocities,
                view.nObjects, nthreads, 0, stream,
                view, translation, forces.devPtr() );
    }
}
   
void PinObjectPlugin::afterIntegration(cudaStream_t stream)
{
    // If the object IS rigid, modify forces and torques
    if (rov != nullptr)
    {
        debug("Restricting rigid motion of OV '%s' as per plugin '%s'", ovName.c_str(), name.c_str());

        const int nthreads = 32;
        ROVviewWithOldMotion view(rov, rov->local());
        SAFE_KERNEL_LAUNCH(
                restrictRigidMotion,
                getNblocks(view.nObjects, nthreads), nthreads, 0, stream,
                view, translation, rotation, sim->getCurrentDt(),
                forces.devPtr(), torques.devPtr() );
    }
}

void PinObjectPlugin::serializeAndSend(cudaStream_t stream)
{
    count++;
    if (count % reportEvery != 0) return;

    forces.downloadFromDevice(stream);
    if (rov != nullptr)
        torques.downloadFromDevice(stream);

    SimpleSerializer::serialize(sendBuffer, currentTime, reportEvery, forces, torques);
    send(sendBuffer);

    forces.clearDevice(stream);
    torques.clearDevice(stream);
}


ReportPinObjectPlugin::ReportPinObjectPlugin(std::string name, std::string path) :
                PostprocessPlugin(name), path(path)
{    }

void ReportPinObjectPlugin::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void ReportPinObjectPlugin::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string ovName;
    SimpleSerializer::deserialize(data, ovName);
    if (activated && rank == 0)
        fout = fopen( (path + "/" + ovName + ".txt").c_str(), "w" );
}

void ReportPinObjectPlugin::deserialize(MPI_Status& stat)
{
    std::vector<float4> forces, torques;
    float currentTime;
    int nsamples;

    SimpleSerializer::deserialize(data, currentTime, nsamples, forces, torques);

    MPI_Check( MPI_Reduce( (rank == 0 ? MPI_IN_PLACE : forces.data()),  forces.data(),  forces.size()*4,  MPI_FLOAT, MPI_SUM, 0, comm) );
    MPI_Check( MPI_Reduce( (rank == 0 ? MPI_IN_PLACE : torques.data()), torques.data(), torques.size()*4, MPI_FLOAT, MPI_SUM, 0, comm) );

    if (activated && rank == 0)
    {
        for (int i=0; i < forces.size(); i++)
        {
            forces[i] /= nsamples;
            fprintf(fout, "%d  %f  %f %f %f", i, currentTime, forces[i].x, forces[i].y, forces[i].z);

            if (i < torques.size())
            {
                torques[i] /= nsamples;
                fprintf(fout, "  %f %f %f", torques[i].x, torques[i].y, torques[i].z);
            }

            fprintf(fout, "\n");
        }

        fflush(fout);
    }
}

