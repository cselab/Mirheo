#include "dump_obj_position.h"
#include "simple_serializer.h"
#include "utils.h"

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

void ObjPositionsPlugin::beforeForces(cudaStream_t stream)
{
	if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

	ov->local()->extraPerObject.getData<int>("ids")->downloadFromDevice(stream);
	ov->local()->extraPerObject.getData<LocalObjectVector::COMandExtent> ("com_extents")->downloadFromDevice(stream);

	if (ov->local()->extraPerObject.checkChannelExists("motions"))
		ov->local()->extraPerObject.getData<RigidMotion> ("motions")->downloadFromDevice(stream);
}

void ObjPositionsPlugin::serializeAndSend(cudaStream_t stream)
{
	if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

	debug2("Plugin %s is sending now data", name.c_str());

	PinnedBuffer<RigidMotion> dummy(0);

	std::vector<char> data;
	SimpleSerializer::serialize(data,
			currentTime,
			ov->name,
			ov->domain,
			*ov->local()->extraPerObject.getData<int>("ids"),
			*ov->local()->extraPerObject.getData<LocalObjectVector::COMandExtent>("com_extents"),
			ov->local()->extraPerObject.checkChannelExists("motions") ?
					*ov->local()->extraPerObject.getData<RigidMotion>("motions") : dummy );

	send(data);
}

//=================================================================================

void writePositions(MPI_Comm comm, DomainInfo domain, std::string fname, float curTime, std::vector<int>& ids,
		std::vector<LocalObjectVector::COMandExtent> coms, std::vector<RigidMotion> motions)
{
	int rank;
	MPI_Check( MPI_Comm_rank(comm, &rank) );

	int np = ids.size();
	int n = np;
	MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );

	MPI_File f;
	MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE|MPI_MODE_DELETE_ON_CLOSE|MPI_MODE_WRONLY, MPI_INFO_NULL, &f) );
	MPI_Check( MPI_File_close(&f) );
	MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f) );

	std::stringstream ss;
	ss.setf(std::ios::fixed, std::ios::floatfield);
	ss.precision(5);

//	if (rank == 0)
//	{
//		ss <<  n << "\n";
//		ss << "# created by uDeviceX" << "\n";
//
//		info("Object position dump to %s: total number of particles: %d", fname.c_str(), n);
//	}

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
					<< std::setw(10) << motion.torque.z << std::endl;
		}
	}

	std::string content = ss.str();

	MPI_Offset len = content.size();
	MPI_Offset offset = 0;
	MPI_Check( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm));

	MPI_Status status;
	MPI_Check( MPI_File_write_at_all(f, offset, content.c_str(), len, MPI_CHAR, &status) );
	MPI_Check( MPI_File_close(&f));

	// Sort contents w.r.t. ids
	std::string command = "sort -n " + fname + " -o " + fname;
	if (rank == 0)
	{
		if ( system(command.c_str()) != 0 )
			error("Could not sort file '%s'", fname.c_str());
	}
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

void ObjPositionsDumper::deserialize(MPI_Status& stat)
{
	float curTime;
	std::string ovName;
	DomainInfo domain;
	std::vector<int> ids;
	std::vector<LocalObjectVector::COMandExtent> coms;
	std::vector<RigidMotion> motions;

	SimpleSerializer::deserialize(data, curTime, ovName, domain, ids, coms, motions);

	std::string tstr = std::to_string(timeStamp++);
	std::string currentFname = path + "/" + ovName + "_" + std::string(5 - tstr.length(), '0') + tstr + ".txt";

	if (activated)
		writePositions(comm, domain, currentFname, curTime, ids, coms, motions);
}



