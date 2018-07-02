#include "dump_obj_position.h"
#include "simple_serializer.h"
#include <core/utils/folders.h>

#include <core/simulation.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>

#include <regex>

ObjPositionsPlugin::ObjPositionsPlugin(std::string name, std::string ovName, int dumpEvery) :
	SimulationPlugin(name), ovName(ovName),
	dumpEvery(dumpEvery)
{	}

void ObjPositionsPlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
	SimulationPlugin::setup(sim, comm, interComm);

	ov = dynamic_cast<ObjectVector*>(sim->getPVbyName(ovName));
	if (ov == nullptr)
		die("No such object vector registered: %s", ovName.c_str());

	info("Plugin %s initialized for the following object vectors: %s", name.c_str(), ovName.c_str());
}

void ObjPositionsPlugin::handshake()
{
	SimpleSerializer::serialize(sendBuffer, ovName);
	send(sendBuffer);
}

void ObjPositionsPlugin::beforeForces(cudaStream_t stream)
{
	if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

	ov->local()->extraPerObject.getData<int>("ids")->downloadFromDevice(stream);
	ov->local()->extraPerObject.getData<LocalObjectVector::COMandExtent> ("com_extents")->downloadFromDevice(stream);

	if (ov->local()->extraPerObject.checkChannelExists("old_motions"))
		ov->local()->extraPerObject.getData<RigidMotion> ("old_motions")->downloadFromDevice(stream);
}

void ObjPositionsPlugin::serializeAndSend(cudaStream_t stream)
{
	if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

	debug2("Plugin %s is sending now data", name.c_str());

	PinnedBuffer<RigidMotion> dummy(0);

	SimpleSerializer::serialize(sendBuffer,
			currentTime,
			ov->domain,
			*ov->local()->extraPerObject.getData<int>("ids"),
			*ov->local()->extraPerObject.getData<LocalObjectVector::COMandExtent>("com_extents"),
			ov->local()->extraPerObject.checkChannelExists("old_motions") ?
					*ov->local()->extraPerObject.getData<RigidMotion>("old_motions") : dummy );

	send(sendBuffer);
}

//=================================================================================

void writePositions(MPI_Comm comm, DomainInfo domain, MPI_File& fout, float curTime, std::vector<int>& ids,
		std::vector<LocalObjectVector::COMandExtent> coms, std::vector<RigidMotion> motions)
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

		if (i < motions.size())
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

	MPI_Offset len = content.size();
	MPI_Check( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm) );

	MPI_Status status;
	MPI_Check( MPI_File_write_at_all(fout, offset + size, content.c_str(), len, MPI_CHAR, &status) );
}

//=================================================================================


ObjPositionsDumper::ObjPositionsDumper(std::string name, std::string path) :
		PostprocessPlugin(name), path(path)
{	}

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
	float curTime;
	DomainInfo domain;
	std::vector<int> ids;
	std::vector<LocalObjectVector::COMandExtent> coms;
	std::vector<RigidMotion> motions;

	SimpleSerializer::deserialize(data, curTime, domain, ids, coms, motions);

	if (activated)
		writePositions(comm, domain, fout, curTime, ids, coms, motions);
}



