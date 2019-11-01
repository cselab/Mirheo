#include "dump_obj_stats.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace ObjStatsPluginKernels
{

__global__ void collectObjStats(OVview view, RigidMotion *motionStats)
{
    const int objId  = blockIdx.x;
    const int tid    = threadIdx.x;
    const int laneId = tid % warpSize;

    RigidMotion local = {0};

    const auto com = view.comAndExtents[objId].com;
    
    for (int i = tid; i < view.objSize; i += blockDim.x)
    {
        int pid = objId * view.objSize + i;
        const Particle p = view.readParticle(pid);
        real3 f = make_real3(view.forces[pid]);

        real3 dr = p.r - com;
        
        local.vel    += p.u;
        local.omega  += cross(dr, p.u);
        local.force  += f;
        local.torque += cross(dr, f);
    }

    auto add = [](const RigidReal& a, const RigidReal& b) {return a+b;};

    warpReduce(local.vel,    add);
    warpReduce(local.omega,  add);
    warpReduce(local.force,  add);
    warpReduce(local.torque, add);

    if (laneId == 0)
    {
        atomicAdd( &motionStats[objId].vel,   local.vel   / view.objSize);
        atomicAdd( &motionStats[objId].omega, local.omega / view.objSize);

        atomicAdd( &motionStats[objId].force,  local.force );
        atomicAdd( &motionStats[objId].torque, local.torque);
    }
}

} // namespace ObjStatsPluginKernels

ObjStatsPlugin::ObjStatsPlugin(const MirState *state, std::string name, std::string ovName, int dumpEvery) :
    SimulationPlugin(state, name),
    ovName(ovName),
    dumpEvery(dumpEvery)
{}

void ObjStatsPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    ov = dynamic_cast<ObjectVector*>(simulation->getPVbyName(ovName));
    if (ov == nullptr)
        die("No such object vector registered: %s", ovName.c_str());

    info("Plugin %s initialized for the following object vectors: %s", name.c_str(), ovName.c_str());
}

void ObjStatsPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer, ovName);
    send(sendBuffer);
}

void ObjStatsPlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    ids.copy(  *ov->local()->dataPerObject.getData<int64_t>(ChannelNames::globalIds), stream);
    coms.copy( *ov->local()->dataPerObject.getData<COMandExtent>(ChannelNames::comExtents), stream);

    if (ov->local()->dataPerObject.checkChannelExists(ChannelNames::oldMotions))
    {
        auto& oldMotions = *ov->local()->dataPerObject.getData<RigidMotion> (ChannelNames::oldMotions);
        motions.copy(oldMotions, stream);
        isRov = true;
    }
    else
    {
        const int nthreads = 128;
        OVview view(ov, ov->local());
        motionStats.resize_anew(view.nObjects);

        motionStats.clear(stream);

        SAFE_KERNEL_LAUNCH(
            ObjStatsPluginKernels::collectObjStats,
            view.nObjects, nthreads, 0, stream,
            view, motionStats.devPtr());

        motions.copy(motionStats, stream);
        isRov = false;
    }
    
    savedTime = state->currentTime;
    needToSend = true;
}

void ObjStatsPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend) return;

    debug2("Plugin %s is sending now data", name.c_str());

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, savedTime, state->domain, isRov, ids, coms, motions);
    send(sendBuffer);
    
    needToSend=false;
}

//=================================================================================

static void writeStats(MPI_Comm comm, DomainInfo domain, MPI_File& fout, real curTime, const std::vector<int64_t>& ids,
                       const std::vector<COMandExtent>& coms, const std::vector<RigidMotion>& motions, bool isRov)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    int np = ids.size();
    int n = np;
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );

    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(5);

    for (int i = 0; i < np; ++i)
    {
        auto com = coms[i];
        com.com = domain.local2global(com.com);

        ss << ids[i] << " " << curTime << "   "
                << std::setw(10) << com.com.x << " "
                << std::setw(10) << com.com.y << " "
                << std::setw(10) << com.com.z;

        const auto& motion = motions[i];

        if (isRov)
        {
            ss << "    "
               << std::setw(10) << motion.q.x << " "
               << std::setw(10) << motion.q.y << " "
               << std::setw(10) << motion.q.z << " "
               << std::setw(10) << motion.q.w;
        }

        ss << "    "   
           << std::setw(10) << motion.vel.x << " "
           << std::setw(10) << motion.vel.y << " "
           << std::setw(10) << motion.vel.z << "    "
            
           << std::setw(10) << motion.omega.x << " "
           << std::setw(10) << motion.omega.y << " "
           << std::setw(10) << motion.omega.z << "    "
            
           << std::setw(10) << motion.force.x << " "
           << std::setw(10) << motion.force.y << " "
           << std::setw(10) << motion.force.z << "    "
            
           << std::setw(10) << motion.torque.x << " "
           << std::setw(10) << motion.torque.y << " "
           << std::setw(10) << motion.torque.z;

        ss << std::endl;
    }

    std::string content = ss.str();

    MPI_Offset offset = 0, size;
    MPI_Check( MPI_File_get_size(fout, &size) );
    MPI_Check( MPI_Barrier(comm) );

    MPI_Offset len = content.size();
    MPI_Check( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm) );

    MPI_Status status;
    MPI_Check( MPI_File_write_at_all(fout, offset + size, content.c_str(), len, MPI_CHAR, &status) );
    MPI_Check( MPI_Barrier(comm) );
}

//=================================================================================


ObjStatsDumper::ObjStatsDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(makePath(path))
{}

ObjStatsDumper::~ObjStatsDumper()
{
    if (activated)
        MPI_Check( MPI_File_close(&fout) );
}

void ObjStatsDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void ObjStatsDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string ovName;
    SimpleSerializer::deserialize(data, ovName);

    if (activated)
    {
        auto fname = path + ovName + ".txt";
        MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout) );
        MPI_Check( MPI_File_close(&fout) );
        MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fout) );
    }
}


void ObjStatsDumper::deserialize()
{
    MirState::TimeType curTime;
    DomainInfo domain;
    std::vector<int64_t> ids;
    std::vector<COMandExtent> coms;
    std::vector<RigidMotion> motions;
    bool isRov;

    SimpleSerializer::deserialize(data, curTime, domain, isRov, ids, coms, motions);

    if (activated)
        writeStats(comm, domain, fout, curTime, ids, coms, motions, isRov);
}



