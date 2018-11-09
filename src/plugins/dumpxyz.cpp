#include "dumpxyz.h"
#include "simple_serializer.h"
#include <core/utils/folders.h>

#include <core/simulation.h>
#include <core/pvs/particle_vector.h>
#include "utils/xyz.h"


XYZPlugin::XYZPlugin(std::string name, std::string pvName, int dumpEvery) :
    SimulationPlugin(name), pvName(pvName),
    dumpEvery(dumpEvery)
{ }

void XYZPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    info("Plugin %s initialized for the following particle vector: %s", name.c_str(), pvName.c_str());
}

void XYZPlugin::beforeForces(cudaStream_t stream)
{
    if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

    downloaded.copy(pv->local()->coosvels, stream);
}

void XYZPlugin::serializeAndSend(cudaStream_t stream)
{
    if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;

    debug2("Plugin %s is sending now data", name.c_str());

    for (auto& p : downloaded)
        p.r = simulation->domain.local2global(p.r);

    waitPrevSend();
    SimpleSerializer::serialize(data, pv->name, downloaded);
    send(data);
}

//=================================================================================

XYZDumper::XYZDumper(std::string name, std::string path) :
        PostprocessPlugin(name), path(path)
{    }

void XYZDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void XYZDumper::deserialize(MPI_Status& stat)
{
    std::string pvName;

    SimpleSerializer::deserialize(data, pvName, ps);

    std::string tstr = std::to_string(timeStamp++);
    std::string currentFname = path + "/" + pvName + "_" + std::string(5 - tstr.length(), '0') + tstr + ".xyz";

    if (activated)
        writeXYZ(comm, currentFname, ps.data(), ps.size());
}



