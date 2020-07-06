// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "dump_obj_stats.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/path.h>
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
    isRov_ = dynamic_cast<RigidObjectVector*>(ov_) != nullptr;
    hasTypeIds_ = ov_->local()->dataPerObject.checkChannelExists(channel_names::membraneTypeId);
    info("Plugin '%s' initialized for object vector '%s'", getCName(), ovName_.c_str());
}

void ObjStatsPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer_, ovName_, isRov_, hasTypeIds_);
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
    }

    if (hasTypeIds_)
        typeIds_.copy( *lov->dataPerObject.getData<int>(channel_names::membraneTypeId), stream);

    savedTime_ = getState()->currentTime;
    needToSend_ = true;
}

void ObjStatsPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend_)
        return;

    debug2("Plugin %s is sending now data", getCName());

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, savedTime_, getState()->domain, isRov_, ids_, coms_, motions_, hasTypeIds_, typeIds_);
    _send(sendBuffer_);

    needToSend_=false;
}

//=================================================================================

static void writeHeader(MPI_Comm comm, MPI_File& fout, bool isRov, bool hasTypeIds)
{
    int rank;
    MPI_Check(MPI_Comm_rank(comm, &rank));
    if (rank == 0)
    {
        std::string header = "objId,time,comx,comy,comz";
        if (isRov)
            header += ",qw,qx,qy,qz";
        header += ",vx,vy,vz,wx,wy,wz,fx,fy,fz,Tx,Ty,Tz";
        if (hasTypeIds)
            header += ",typeIds";
        header += '\n';

        MPI_Check( MPI_File_write(fout, header.c_str(), header.size(), MPI_CHAR, MPI_STATUS_IGNORE) );
    }
    MPI_Check( MPI_Barrier(comm) );
}

static void writeStats(MPI_Comm comm, DomainInfo domain, MPI_File& fout, real curTime, const std::vector<int64_t>& ids,
                       const std::vector<COMandExtent>& coms, const std::vector<RigidMotion>& motions, bool isRov,
                       bool hasTypeIds, const std::vector<int>& typeIds)
{
    const int nObjs = ids.size();

    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(5);
    const char sep = ',';

    for (int i = 0; i < nObjs; ++i)
    {
        auto com = coms[i].com;
        com = domain.local2global(com);

        ss << ids[i] << sep << curTime << sep
           << com.x << sep
           << com.y << sep
           << com.z;

        const auto& motion = motions[i];

        if (isRov)
        {
            ss << sep
               << motion.q.w << sep
               << motion.q.x << sep
               << motion.q.y << sep
               << motion.q.z;
        }

        ss << sep
           << motion.vel.x << sep
           << motion.vel.y << sep
           << motion.vel.z << sep

           << motion.omega.x << sep
           << motion.omega.y << sep
           << motion.omega.z << sep

           << motion.force.x << sep
           << motion.force.y << sep
           << motion.force.z << sep

           << motion.torque.x << sep
           << motion.torque.y << sep
           << motion.torque.z;

        if (hasTypeIds)
            ss << sep << typeIds[i];

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
    bool isRov;
    bool hasTypeIds;
    SimpleSerializer::deserialize(data_, ovName, isRov, hasTypeIds);

    if (activated_)
    {
        const std::string fname = joinPaths(path_, setExtensionOrDie(ovName, "csv"));
        MPI_Check( MPI_File_open(comm_, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout_) );
        MPI_Check( MPI_File_close(&fout_) );
        MPI_Check( MPI_File_open(comm_, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fout_) );
        writeHeader(comm_, fout_, isRov, hasTypeIds);
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
