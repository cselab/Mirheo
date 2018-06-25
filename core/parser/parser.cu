#include "parser.h"

#include <core/udevicex.h>
#include <core/simulation.h>
#include <core/postproc.h>

#include <core/xml/pugixml.hpp>
#include <utility>
#include <fstream>
#include <sstream>
#include <regex>

#include "belonging_factory.h"
#include "bouncers_factory.h"
#include "ic_factory.h"
#include "integrators_factory.h"
#include "interactions_factory.h"
#include "plugins_factory.h"
#include "pv_factory.h"
#include "walls_factory.h"

#include <core/utils/make_unique.h>

std::string subsVariables(const std::string& variables, const std::string& content)
{
	std::regex pattern(R"(([^\s,]+)\s*=\s*([^\s,]+))");
	std::smatch match;
	std::string result = content;

	auto searchStart = variables.cbegin();

	while ( std::regex_search( searchStart, variables.cend(), match, pattern ) )
	{
		auto varname = match[1].str();
		auto value   = match[2].str();

		searchStart += match.position() + match.length();

		printf("regex : '%s'\n", (R"(\$)" + varname + R"(\b|\$\{)" + varname + R"(\})").c_str());
		// \$name\b|\$\{$name\}
		std::regex subs(R"(\$)" + varname + R"(\b|\$\{)" + varname + R"(\})");
		result = std::regex_replace(result, subs, value);
	}

	return result;
}

//================================================================================================
// Main parser
//================================================================================================

Parser::Parser(std::string xmlname, int forceDebugLvl, std::string variables) :
		forceDebugLvl(forceDebugLvl)
{
	// Read the file into memory
	std::ifstream f(xmlname);
	if (!f.good()) // Can't die here, logger is not yet setup
	{
		fprintf(stderr, "Couldn't open script file (not found or not accessible)\n");
		exit(1);
	}

	std::stringstream buffer;
	buffer << f.rdbuf();

	auto result = config.load( subsVariables(variables, buffer.str()).c_str() );
	if (!result)
	{
		fprintf(stderr, "Couldn't open script file, xml parser says: \"%s\"\n", result.description());
		exit(1);
	}
}

int Parser::getNIterations()
{
	auto simNode = config.child("simulation");
	if (simNode.type() == pugi::node_null)
		die("Simulation is not defined");

	return simNode.child("run").attribute("niters").as_int(1);
}

std::unique_ptr<uDeviceX> Parser::setup_uDeviceX(Logger& logger, bool useGpuAwareMPI)
{
	auto simNode = config.child("simulation");
	if (simNode.type() == pugi::node_null)
		die("Simulation is not defined");

	// A few global simulation parameters
	std::string name = simNode.attribute("name").as_string();
	std::string logname = simNode.attribute("logfile").as_string(name.c_str());
	float3 globalDomainSize = simNode.child("domain").attribute("size").as_float3({32, 32, 32});

	int3 nranks3D = simNode.attribute("mpi_ranks").as_int3({1, 1, 1});
	int debugLvl  = simNode.attribute("debug_lvl").as_int(2);
	if (forceDebugLvl >= 0) debugLvl = forceDebugLvl;

	auto udx = std::make_unique<uDeviceX> (nranks3D, globalDomainSize, logger, logname, debugLvl, useGpuAwareMPI);

	if (udx->isComputeTask())
	{
		for (auto node : simNode.children())
		{
			if ( std::string(node.name()) == "particle_vector" )
			{
				auto pv = ParticleVectorFactory::create(node);
				auto ic = InitialConditionsFactory::create(node.child("generate"));
				auto checkpointEvery = node.attribute("checkpoint_every").as_int(0);
				udx->sim->registerParticleVector(std::move(pv), std::move(ic), checkpointEvery);
			}

			if ( std::string(node.name()) == "interaction" )
			{
				auto inter = InteractionFactory::create(node);
				auto name = inter->name;
				udx->sim->registerInteraction(std::move(inter));

				for (auto apply_to : node.children("apply_to"))
					udx->sim->setInteraction(name,
							apply_to.attribute("pv1").as_string(),
							apply_to.attribute("pv2").as_string());
			}

			if ( std::string(node.name()) == "integrator" )
			{
				auto integrator = IntegratorFactory::create(node);
				auto name = integrator->name;
				udx->sim->registerIntegrator(std::move(integrator));

				for (auto apply_to : node.children("apply_to"))
					udx->sim->setIntegrator(name, apply_to.attribute("pv").as_string());
			}

			if ( std::string(node.name()) == "wall" )
			{
				auto wall = WallFactory::create(node);
				auto name = wall->name;
				udx->sim->registerWall(std::move(wall), node.attribute("check_every").as_int(0));

				for (auto apply_to : node.children("apply_to"))
					udx->sim->setWallBounce(name, apply_to.attribute("pv").as_string());
			}

			if ( std::string(node.name()) == "object_bouncer" )
			{
				auto bouncer = BouncerFactory::create(node);
				auto name = bouncer->name;
				udx->sim->registerBouncer(std::move(bouncer));

				// TODO do this normal'no epta
				for (auto apply_to : node.children("apply_to"))
					udx->sim->setBouncer(name,
							node.attribute("ov").as_string(),
							apply_to.attribute("pv").as_string());
			}

			if ( std::string(node.name()) == "object_belonging_checker" )
			{
				auto checker = ObjectBelongingCheckerFactory::create(node);
				auto name = checker->name;
				udx->sim->registerObjectBelongingChecker(std::move(checker));
				udx->sim->setObjectBelongingChecker(name, node.attribute("object_vector").as_string());

				for (auto apply_to : node.children("apply_to"))
				{
					std::string source  = apply_to.attribute("pv"). as_string();
					std::string inside  = apply_to.attribute("inside"). as_string();
					std::string outside = apply_to.attribute("outside").as_string();
					auto checkEvery     = apply_to.attribute("check_every").as_int(0);

					udx->sim->applyObjectBelongingChecker(name, source, inside, outside, checkEvery);
				}
			}
		}
	}

	for (auto node : simNode.children())
		if ( std::string(node.name()) == "plugin" )
			udx->registerPlugins( std::move(PluginFactory::create(node, udx->isComputeTask())) );

	// Write the script that we actually execute (with substitutions)
	int rank;
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	if (rank == 0)
	{
		config.save_file("uDeviceX_input_script.xml");
	}

	return udx;
}






