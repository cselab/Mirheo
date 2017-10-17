#include "dumpxyz.h"
#include "simple_serializer.h"
#include "string2vector.h"

#include <core/simulation.h>
#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>

#include <regex>

XYZPlugin::XYZPlugin(std::string name, std::string pvName, int dumpEvery) :
	SimulationPlugin(name, true), pvName(pvName),
	dumpEvery(dumpEvery)
{ }

void XYZPlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
	SimulationPlugin::setup(sim, comm, interComm);

	pv = sim->getPVbyName(pvName);
	if (pv == nullptr)
		die("No such particle vector registered: %s", pvName.c_str());

	info("Plugin %s initialized for the following particle vector: %s", name.c_str(), pvName.c_str());
}

void XYZPlugin::beforeForces(cudaStream_t stream)
{
	if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

	pv->local()->coosvels.downloadFromDevice(stream);
}

void XYZPlugin::serializeAndSend(cudaStream_t stream)
{
	if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

	debug2("Plugin %s is sending now data", name.c_str());

	for (int i=0; i < pv->local()->size(); i++)
		pv->local()->coosvels[i].r = pv->domain.local2global(pv->local()->coosvels[i].r);

	send(pv->local()->coosvels.hostPtr(), pv->local()->coosvels.size() * sizeof(Particle));
}

//=================================================================================

void writeXYZ(MPI_Comm comm, std::string fname, Particle* particles, int np)
{
	int rank;
	MPI_Check( MPI_Comm_rank(comm, &rank) );

	int n = np;
	MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );

	MPI_File f;
	MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE|MPI_MODE_DELETE_ON_CLOSE|MPI_MODE_WRONLY, MPI_INFO_NULL, &f) );
	MPI_Check( MPI_File_close(&f) );
	MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f) );

	std::stringstream ss;
	ss.setf(std::ios::fixed, std::ios::floatfield);
	ss.precision(5);

	if (rank == 0)
	{
		ss <<  n << "\n";
		ss << "# created by uDeviceX" << "\n";

		info("xyz dump to %s: total number of particles: %d", fname.c_str(), n);
	}

	for(int i = 0; i < np; ++i)
	{
		Particle& p = particles[i];

		ss << rank << " "
				<< std::setw(10) << p.r.x << " "
				<< std::setw(10) << p.r.y << " "
				<< std::setw(10) << p.r.z << "\n";
	}

	std::string content = ss.str();

	MPI_Offset len = content.size();
	MPI_Offset offset = 0;
	MPI_Check( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm));

	MPI_Status status;
	MPI_Check( MPI_File_write_at_all(f, offset, content.c_str(), len, MPI_CHAR, &status) );
	MPI_Check( MPI_File_close(&f));
}

XYZDumper::XYZDumper(std::string name, std::string path) :
		PostprocessPlugin(name), path(path)
{	}

void XYZDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
	PostprocessPlugin::setup(comm, interComm);

	int rank;
	MPI_Check( MPI_Comm_rank(comm, &rank) );

	std::regex re(R".(^(.*/)(.+)).");
	std::smatch match;
	if (std::regex_match(path, match, re))
	{
		std::string folders  = match[1].str();
		std::string command = "mkdir -p " + folders;
		if (rank == 0)
		{
			if ( system(command.c_str()) != 0 )
			{
				error("Could not create folders or files by given path, dumping will be disabled.");
				activated = false;
			}
		}
	}
}

void XYZDumper::deserialize(MPI_Status& stat)
{
	int np = size / sizeof(Particle);

	std::string tstr = std::to_string(timeStamp++);
	std::string currentFname = path + std::string(5 - tstr.length(), '0') + tstr + ".xyz";

	if (activated)
		writeXYZ(comm, currentFname, (Particle*)data.data(), np);
}



