#include <string>

#include <core/simulation.h>
#include "freeze_particles.h"
#include <core/interactions/sampler.h>

#include <core/argument_parser.h>

Logger logger;

void writeXYZ(MPI_Comm comm, std::string fname, ParticleVector* pv)
{
	int rank;
	MPI_Check( MPI_Comm_rank(comm, &rank) );

	int dims[3], periods[3], coords[3];
	MPI_Check( MPI_Cart_get(comm, 3, dims, periods, coords) );

	const int nlocal = pv->local()->size();
	int n = nlocal;
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
		ss << pv->name << "\n";

		info("xyz dump of %s: total number of particles: %d", pv->name.c_str(), n);
	}

	pv->local()->coosvels.downloadFromDevice(0);
	for(int i = 0; i < nlocal; ++i)
	{
		Particle p = pv->local()->coosvels[i];
		p.r = pv->local2global(p.r);

		ss << rank << " "
				<< std::setw(10) << p.r.x << " "
				<< std::setw(10) << p.r.y << " "
				<< std::setw(10) << p.r.z << "\n";
	}

	string content = ss.str();

	MPI_Offset len = content.size();
	MPI_Offset offset = 0;
	MPI_Check( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm));

	MPI_Status status;
	MPI_Check( MPI_File_write_at_all(f, offset, content.c_str(), len, MPI_CHAR, &status) );
	MPI_Check( MPI_File_close(&f));
}

int main(int argc, char** argv)
{
	srand48(0);

	std::string xmlname, wname;
	int nepochs;

	{
		using namespace ArgumentParser;

		std::vector<OptionStruct> opts
		({
			{'i', "input",  STRING, "Input script",                &xmlname,   std::string("script.xml")},
			{'n', "name",   STRING, "Name of the wall to process", &wname,     std::string("wall")},
			{'e', "epochs", INT,    "Number of sampling epochs",   &nepochs,   100}
		});

		Parser parser(opts);
		parser.parse(argc, argv);
	}

	MPI_Init(&argc, &argv);
	logger.init(MPI_COMM_WORLD, "genwall.log", 9);

	pugi::xml_document config;
	pugi::xml_parse_result result = config.load_file(xmlname.c_str());
	if (!result)
		die("Couldn't open script file, parser says: \"%s\"", result.description());

	pugi::xml_node wallXML;
	for (auto node : config.child("simulation").children("wall"))
	{
		if ( std::string(node.attribute("name").as_string()) == wname )
			wallXML = node;
	}

	if (wallXML.type() == pugi::node_null)
		die("Wall %s was not found in the script", wname.c_str());


	float3 globalDomainSize = config.child("simulation").child("domain").attribute("size").as_float3({32, 32, 32});
	int3 nranks3D = config.child("simulation").attribute("mpi_ranks").as_int3({1, 1, 1});

	Simulation* sim = new Simulation(nranks3D, globalDomainSize, MPI_COMM_WORLD, MPI_COMM_NULL);

	ParticleVector *startingPV = new ParticleVector("starting");
	ParticleVector *wallPV     = new ParticleVector("wall");
	ParticleVector *final      = new ParticleVector("final");
	InitialConditions* ic = new UniformIC(wallXML.child("generate"));
	InitialConditions* dummyIC = new DummyIC();

	Wall* wall = new Wall(
			wallXML.attribute("name").as_string(),
			wallXML.attribute("file_name").as_string(),
			wallXML.attribute("h").as_float3({0.25, 0.25, 0.25}));

	// Generate pv, but don't register it
	ic->exec(sim->getCartComm(), startingPV, sim->globalDomainStart, sim->localDomainSize, 0);

	// Register and create sdf
	sim->registerWall(wall, false);
	// Produce new pv out of particles inside the wall
	freezeParticlesInWall(wall, startingPV, wallPV, -3, 4);
	sim->registerParticleVector(wallPV, dummyIC);

	Interaction* sampler = new MCMCSampler(wallXML.child("generate"), wall, -3, 4);
	sampler->name = "sampler";
	sim->registerInteraction(sampler);
	sim->setInteraction("wall", "wall", "sampler");

	sim->init();
	sim->run(nepochs);

	freezeParticlesInWall(wall, wallPV, final, 0, 1.2);

	writeXYZ(sim->getCartComm(), "wall.xyz", wallPV);
	writeXYZ(sim->getCartComm(), "final.xyz", final);
	final->checkpoint(sim->getCartComm(), "./");

	sim->finalize();
}
