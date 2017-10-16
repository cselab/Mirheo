#include <string>
#include <utility>

#include <core/simulation.h>
#include "freeze_particles.h"
#include <core/interactions/sampler.h>
#include <core/initial_conditions/dummy.h>
#include <core/initial_conditions/uniform.h>

#include <core/argument_parser.h>

#include <core/walls/simple_stationary_wall.h>

#include <core/walls/stationary_walls/cylinder.h>
#include <core/walls/stationary_walls/sphere.h>
#include <core/walls/stationary_walls/sdf.h>

// This should be taken from parser
class WallFactory
{
private:
	static Wall* createSphereWall(pugi::xml_node node)
	{
		auto name   = node.attribute("name").as_string("");

		auto center = node.attribute("center").as_float3();
		auto radius = node.attribute("radius").as_float(1);
		auto inside = node.attribute("inside").as_bool(false);

		StationaryWall_Sphere sphere(center, radius, inside);

		return (Wall*) new SimpleStationaryWall<StationaryWall_Sphere>(name, std::move(sphere));
	}

	static Wall* createCylinderWall(pugi::xml_node node)
	{
		auto name   = node.attribute("name").as_string("");

		auto center = node.attribute("center").as_float2();
		auto radius = node.attribute("radius").as_float(1);
		auto inside = node.attribute("inside").as_bool(false);

		std::string dirStr = node.attribute("axis").as_string("x");

		StationaryWall_Cylinder::Direction dir;
		if (dirStr == "x") dir = StationaryWall_Cylinder::Direction::x;
		if (dirStr == "y") dir = StationaryWall_Cylinder::Direction::y;
		if (dirStr == "z") dir = StationaryWall_Cylinder::Direction::z;

		StationaryWall_Cylinder cylinder(center, radius, dir, inside);

		return (Wall*) new SimpleStationaryWall<StationaryWall_Cylinder>(name, std::move(cylinder));
	}

	static Wall* createSDFWall(pugi::xml_node node)
	{
		auto name    = node.attribute("name").as_string("");

		auto sdfFile = node.attribute("sdf_filename").as_string("wall.sdf");
		auto sdfH    = node.attribute("sdf_h").as_float3( make_float3(0.25f) );

		StationaryWall_SDF sdf(sdfFile, sdfH);

		return (Wall*) new SimpleStationaryWall<StationaryWall_SDF>(name, std::move(sdf));
	}

public:
	static Wall* create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "cylinder")
			return createCylinderWall(node);
		if (type == "sphere")
			return createSphereWall(node);
		if (type == "sdf")
			return createSDFWall(node);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return nullptr;
	}
};



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

	PVview view(pv, pv->local());
	pv->local()->coosvels.downloadFromDevice(0);
	for(int i = 0; i < nlocal; ++i)
	{
		Particle p = pv->local()->coosvels[i];
		p.r = view.local2global(p.r);

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


template<class InsideWallChecker>
static Interaction* createMCMCSampler(pugi::xml_node node, const InsideWallChecker& insideWallChecker)
{
	auto name = node.attribute("name").as_string("");
	auto rc   = node.attribute("rc").as_float(1.0f);

	auto a     = node.attribute("a")    .as_float(50);
	auto kbT   = node.attribute("kbt")  .as_float(1.0);
	auto power = node.attribute("power").as_float(1.0f);

	return (Interaction*) new MCMCSampler<InsideWallChecker>(
			name, rc, a, kbT, power, insideWallChecker );
}


int main(int argc, char** argv)
{
	srand48(0);

	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::string xmlname, wname;
	int nepochs;
	bool needXYZ;

	{
		using namespace ArgumentParser;

		std::vector<OptionStruct> opts
		({
			{'i', "input",  STRING, "Input script",                &xmlname,   std::string("script.xml")},
			{'n', "name",   STRING, "Name of the wall to process", &wname,     std::string("wall")},
			{'e', "epochs", INT,    "Number of sampling epochs",   &nepochs,   50},
			{'x', "xyz",    BOOL,   "Also dump .xyz files",        &needXYZ,   false}
		});

		ArgumentParser::Parser parser(opts, rank == 0);
		parser.parse(argc, argv);
	}

	logger.init(MPI_COMM_WORLD, "genwall.log", 9);


	pugi::xml_document config;
	pugi::xml_parse_result result = config.load_file(xmlname.c_str());
	if (!result)
		die("Couldn't open script file, parser says: \"%s\"", result.description());

	pugi::xml_node wallNode, wallGenNode;
	for (auto node : config.child("simulation").children("wall"))
	{
		if ( std::string(node.attribute("name").as_string()) == wname )
		{
			wallNode = node;
			wallGenNode = node.child("generate_frozen");
		}
	}

	if (wallNode.type() == pugi::node_null)
		die("Wall %s was not found in the script", wname.c_str());
	if (wallGenNode.type() == pugi::node_null)
		die("Wall %s has no generation instructions", wname.c_str());


	float3 globalDomainSize = config.child("simulation").child("domain").attribute("size").as_float3({32, 32, 32});
	int3 nranks3D = config.child("simulation").attribute("mpi_ranks").as_int3({1, 1, 1});

	Simulation* sim = new Simulation(nranks3D, globalDomainSize, MPI_COMM_WORLD, MPI_COMM_NULL);

	ParticleVector *startingPV = new ParticleVector("starting", 1.0);
	ParticleVector *wallPV     = new ParticleVector("wall", 1.0);
	ParticleVector *final      = new ParticleVector(wallGenNode.attribute("name").as_string("final"), 1.0);
	InitialConditions* ic      = new UniformIC(wallGenNode.attribute("density").as_float(4));
	InitialConditions* dummyIC = new DummyIC();

	// Generate pv, but don't register it
	ic->exec(sim->getCartComm(), startingPV, sim->globalDomainStart, sim->localDomainSize, 0);

	// Create and setup wall
	auto wall = WallFactory::create(wallNode);
	sim->registerWall(wall);

	// Produce new pv out of particles inside the wall
	freezeParticlesWrapper(wall, startingPV, wallPV, -3, 4);
	sim->registerParticleVector(wallPV, dummyIC);

	sim->init();
	sim->run(nepochs);

	freezeParticlesWrapper(wall, wallPV, final, 0, 1.2);

	if (needXYZ)
	{
		writeXYZ(sim->getCartComm(), "wall.xyz", wallPV);
		writeXYZ(sim->getCartComm(), final->name+".xyz", final);
	}

	std::string path = wallGenNode.attribute("path").as_string("./");
	std::string command = "mkdir -p " + path;
	if (rank == 0)
	{
		if ( system(command.c_str()) != 0 )
			die("Could not create folders by given path %s", path.c_str());
	}

	final->checkpoint(sim->getCartComm(), path);


	sim->finalize();
}
