#include "dump_obj_position.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/pvs/rigid_object_vector.h>
#include <core/simulation.h>
#include <core/utils/folders.h>

ObjPositionsPlugin::ObjPositionsPlugin(const YmrState *state, std::string name, std::string ovName, int dumpEvery) :
    SimulationPlugin(state, name),
    ovName(ovName),
    dumpEvery(dumpEvery)
{}

void ObjPositionsPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    ov = dynamic_cast<ObjectVector*>(simulation->getPVbyName(ovName));
    if (ov == nullptr)
        die("No such object vector registered: %s", ovName.c_str());

    info("Plugin %s initialized for the following object vectors: %s", name.c_str(), ovName.c_str());
}

void ObjPositionsPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer, ovName);
    send(sendBuffer);
}

void ObjPositionsPlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    ids.copy(  *ov->local()->dataPerObject.getData<int64_t>(ChannelNames::globalIds), stream);
    coms.copy( *ov->local()->dataPerObject.getData<COMandExtent>(ChannelNames::comExtents), stream);

    if (ov->local()->dataPerObject.checkChannelExists(ChannelNames::oldMotions))
        motions.copy( *ov->local()->dataPerObject.getData<RigidMotion> (ChannelNames::oldMotions), stream);
    
    savedTime = state->currentTime;
    needToSend = true;
}

void ObjPositionsPlugin::serializeAndSend(cudaStream_t stream)
{
    if (!needToSend) return;

    debug2("Plugin %s is sending now data", name.c_str());

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, savedTime, state->domain, ids, coms, motions);
    send(sendBuffer);
    
    needToSend=false;
}

//=================================================================================

void writePositions(MPI_Comm comm, DomainInfo domain, MPI_File& fout, float curTime, std::vector<int64_t>& ids,
                    std::vector<COMandExtent> coms, std::vector<RigidMotion> motions)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    int np = ids.size();
    int n = np;
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );

    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(5);

    for(int i = 0; i < np; ++i)
    {
        auto com = coms[i];
        com.com = domain.local2global(com.com);

        ss << ids[i] << " " << curTime << "   "
                << std::setw(10) << com.com.x << " "
                << std::setw(10) << com.com.y << " "
                << std::setw(10) << com.com.z;

        if (!motions.empty())
        {
            auto& motion = motions[i];

            ss << "    "
                    << std::setw(10) << motion.q.x << " "
                    << std::setw(10) << motion.q.y << " "
                    << std::setw(10) << motion.q.z << " "
                    << std::setw(10) << motion.q.w << "    "

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
        }

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


ObjPositionsDumper::ObjPositionsDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(path)
{}

ObjPositionsDumper::~ObjPositionsDumper()
{
    if (activated)
        MPI_Check( MPI_File_close(&fout) );
}

void ObjPositionsDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void ObjPositionsDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string ovName;
    SimpleSerializer::deserialize(data, ovName);

    if (activated)
    {
        auto fname = path + "/" + ovName + ".txt";
        MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout) );
        MPI_Check( MPI_File_close(&fout) );
        MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fout) );
    }
}


void ObjPositionsDumper::deserialize(MPI_Status& stat)
{
    YmrState::TimeType curTime;
    DomainInfo domain;
    std::vector<int64_t> ids;
    std::vector<COMandExtent> coms;
    std::vector<RigidMotion> motions;

    SimpleSerializer::deserialize(data, curTime, domain, ids, coms, motions);

    if (activated)
        writePositions(comm, domain, fout, curTime, ids, coms, motions);
}



