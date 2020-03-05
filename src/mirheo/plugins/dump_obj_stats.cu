#include "dump_obj_stats.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <iomanip>

namespace mirheo
{

namespace obj_stats_plugin_kernels
{

__global__ void collectObjStats(OVview view, RigidMotion *motionStats)
{
    const int objId  = blockIdx.x;
    const int tid    = threadIdx.x;
    const int laneId = tid % warpSize;

    RigidMotion local = {0};

    const real3 com = view.comAndExtents[objId].com;
    
    for (int i = tid; i < view.objSize; i += blockDim.x)
    {
        const int pid = objId * view.objSize + i;
        const Particle p = view.readParticle(pid);
        const real3 f = make_real3(view.forces[pid]);

        const real3 dr = p.r - com;
        
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

} // namespace obj_stats_plugin_kernels

ObjStatsPlugin::ObjStatsPlugin(const MirState *state, std::string name, std::string ovName, int dumpEvery) :
    SimulationPlugin(state, name),
    ovName_(ovName),
    dumpEvery_(dumpEvery)
{}

void ObjStatsPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);
    ov_ = simulation->getOVbyNameOrDie(ovName_);
    info("Plugin '%s' initialized for object vector '%s'", getCName(), ovName_.c_str());
}

void ObjStatsPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer_, ovName_);
    _send(sendBuffer_);
}

void ObjStatsPlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    auto lov = ov_->local();
    
    ids_ .copy( *lov->dataPerObject.getData<int64_t>     (channel_names::globalIds),  stream );
    coms_.copy( *lov->dataPerObject.getData<COMandExtent>(channel_names::comExtents), stream );

    if (auto rov = dynamic_cast<RigidObjectVector*>(ov_))
    {
        auto& oldMotions = *rov->local()->dataPerObject.getData<RigidMotion> (channel_names::oldMotions);
        motions_.copy(oldMotions, stream);
        isRov_ = true;
    }
    else
    {
        const int nthreads = 128;
        OVview view(ov_, lov);
        motionStats_.resize_anew(view.nObjects);

        motionStats_.clear(stream);

        SAFE_KERNEL_LAUNCH(
            obj_stats_plugin_kernels::collectObjStats,
            view.nObjects, nthreads, 0, stream,
            view, motionStats_.devPtr());

        motions_.copy(motionStats_, stream);
        isRov_ = false;
    }

    if (lov->dataPerObject.checkChannelExists(channel_names::membraneTypeId))
    {
        typeIds_.copy( *lov->dataPerObject.getData<int>(channel_names::membraneTypeId), stream);
        hasTypeIds_ = true;
    }
    
    savedTime_ = getState()->currentTime;
    needToSend_ = true;
}

void ObjStatsPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend_) return;

    debug2("Plugin %s is sending now data", getCName());

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, savedTime_, getState()->domain, isRov_, ids_, coms_, motions_, hasTypeIds_, typeIds_);
    _send(sendBuffer_);
    
    needToSend_=false;
}

//=================================================================================

static void writeStats(MPI_Comm comm, DomainInfo domain, MPI_File& fout, real curTime, const std::vector<int64_t>& ids,
                       const std::vector<COMandExtent>& coms, const std::vector<RigidMotion>& motions, bool isRov,
                       bool hasTypeIds, const std::vector<int>& typeIds)
{
    const int np = ids.size();

    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(5);

    for (int i = 0; i < np; ++i)
    {
        auto com = coms[i].com;
        com = domain.local2global(com);

        ss << ids[i] << " " << curTime << "   "
           << std::setw(10) << com.x << " "
           << std::setw(10) << com.y << " "
           << std::setw(10) << com.z;

        const auto& motion = motions[i];

        if (isRov)
        {
            ss << "    "
               << std::setw(10) << motion.q.w << " "
               << std::setw(10) << motion.q.x << " "
               << std::setw(10) << motion.q.y << " "
               << std::setw(10) << motion.q.z;
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

        if (hasTypeIds)
            ss << "    "  << typeIds[i];

        ss << std::endl;
    }

    const std::string content = ss.str();

    MPI_Offset offset = 0, size;
    MPI_Check( MPI_File_get_size(fout, &size) );
    MPI_Check( MPI_Barrier(comm) );

    const MPI_Offset len = content.size();
    MPI_Check( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm) );

    MPI_Check( MPI_File_write_at_all(fout, offset + size, content.c_str(), len, MPI_CHAR, MPI_STATUS_IGNORE) );
    MPI_Check( MPI_Barrier(comm) );
}

//=================================================================================


ObjStatsDumper::ObjStatsDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path))
{}

ObjStatsDumper::~ObjStatsDumper()
{
    if (activated_)
        MPI_Check( MPI_File_close(&fout_) );
}

void ObjStatsDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void ObjStatsDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string ovName;
    SimpleSerializer::deserialize(data_, ovName);

    if (activated_)
    {
        const std::string fname = path_ + ovName + ".txt";
        MPI_Check( MPI_File_open(comm_, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout_) );
        MPI_Check( MPI_File_close(&fout_) );
        MPI_Check( MPI_File_open(comm_, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fout_) );
    }
}


void ObjStatsDumper::deserialize()
{
    MirState::TimeType curTime;
    DomainInfo domain;
    std::vector<int64_t> ids;
    std::vector<COMandExtent> coms;
    std::vector<RigidMotion> motions;
    std::vector<int> typeIds;
    bool isRov;
    bool hasTypeIds;

    SimpleSerializer::deserialize(data_, curTime, domain, isRov, ids, coms, motions, hasTypeIds, typeIds);

    if (activated_)
        writeStats(comm_, domain, fout_, curTime, ids, coms, motions, isRov, hasTypeIds, typeIds);
}

} // namespace mirheo
