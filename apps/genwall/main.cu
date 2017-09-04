#include <string>

#include <core/simulation.h>
#include <core/interactions/sampler.h>

#include "ArgumentParser.h"

Logger logger;

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
			{'e', "epochs", INT,    "Number of sampling epochs",   &nepochs,   10000}
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

	ParticleVector *startingPV = new ParticleVector("starting");
	InitialConditions* ic = new UniformIC(wallXML);

	Wall* wall = new Wall(
			wallXML.attribute("name").as_string(),
			wallXML.attribute("file_name").as_string(),
			wallXML.attribute("h").as_float3({0.25, 0.25, 0.25}), MCMCSampler::minSdf, MCMCSampler::maxSdf);

	Interaction* sampler = new MCMCSampler(wallXML, wall);
	sampler->name = "sampler";

	Simulation* sim = new Simulation(nranks3D, globalDomainSize, MPI_COMM_WORLD, MPI_COMM_NULL);

	sim->registerParticleVector(startingPV, ic);
	sim->registerInteraction(sampler);
	sim->registerWall(wall, "starting", 0.0f);

	sim->setInteraction(wall->name, wall->name, "sampler");

	// Carve the wall
	sim->init();
	sim->run(nepochs);

	wall->getFrozen()->checkpoint(sim->getCartComm(), "./");

	sim->finalize();
}
