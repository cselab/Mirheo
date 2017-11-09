#include "parser.h"

#include <core/udevicex.h>
#include <core/simulation.h>
#include <core/postproc.h>

#include <core/xml/pugixml.hpp>
#include <utility>

#include "belonging_factory.h"
#include "bouncers_factory.h"
#include "ic_factory.h"
#include "integrators_factory.h"
#include "interactions_factory.h"
#include "plugins_factory.h"
#include "pv_factory.h"
#include "walls_factory.h"

//================================================================================================
// Main parser
//================================================================================================

Parser::Parser(std::string xmlname)
{
	pugi::xml_parse_result result = config.load_file(xmlname.c_str());

	if (!result) // Can't die here, logger is not yet setup
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

uDeviceX* Parser::setup_uDeviceX(Logger& logger)
{
	auto simNode = config.child("simulation");
	if (simNode.type() == pugi::node_null)
		die("Simulation is not defined");

	// A few global simulation parameters
	std::string name = simNode.attribute("name").as_string();
	std::string logname = simNode.attribute("logfile").as_string(name.c_str());
	float3 globalDomainSize = simNode.child("domain").attribute("size").as_float3({32, 32, 32});

	int3 nranks3D = simNode.attribute("mpi_ranks").as_int3({1, 1, 1});
	int debugLvl  = simNode.attribute("debug_lvl").as_int(5);

	uDeviceX* udx = new uDeviceX(nranks3D, globalDomainSize, logger, logname, debugLvl);

	if (udx->isComputeTask())
	{
		for (auto node : simNode.children())
		{
			if ( std::string(node.name()) == "particle_vector" )
			{
				auto pv = ParticleVectorFactory::create(node);
				auto ic = InitialConditionsFactory::create(node.child("generate"));
				udx->sim->registerParticleVector(pv, ic);
			}

			if ( std::string(node.name()) == "interaction" )
			{
				auto inter = InteractionFactory::create(node);
				auto name = inter->name;
				udx->sim->registerInteraction(inter);

				for (auto apply_to : node.children("apply_to"))
					udx->sim->setInteraction(name,
							apply_to.attribute("pv1").as_string(),
							apply_to.attribute("pv2").as_string());
			}

			if ( std::string(node.name()) == "integrator" )
			{
				auto integrator = IntegratorFactory::create(node);
				auto name = integrator->name;
				udx->sim->registerIntegrator(integrator);

				for (auto apply_to : node.children("apply_to"))
					udx->sim->setIntegrator(name, apply_to.attribute("pv").as_string());
			}

			if ( std::string(node.name()) == "wall" )
			{
				auto wall = WallFactory::create(node);
				auto name = wall->name;
				udx->sim->registerWall(wall, node.attribute("check_every").as_int(0));

				for (auto apply_to : node.children("apply_to"))
					udx->sim->setWallBounce(name, apply_to.attribute("pv").as_string());
			}

			if ( std::string(node.name()) == "object_bouncer" )
			{
				auto bouncer = BouncerFactory::create(node);
				auto name = bouncer->name;
				udx->sim->registerBouncer(bouncer);

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
				udx->sim->registerObjectBelongingChecker(checker);
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
			udx->registerPlugins( PluginFactory::create(node, udx->isComputeTask()) );

	return udx;
}






