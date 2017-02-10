#include <core/simulation.h>
#include <plugins/plugin.h>
#include <plugins/stats.h>
#include <core/xml/pugixml.hpp>
#include <core/wall.h>

Logger logger;

int main(int argc, char** argv)
{
	pugi::xml_document config;
	pugi::xml_parse_result result = config.load_file("poiseuille.xml");

	float3 globalDomainSize = config.child("simulation").child("domain").attribute("size").as_float3({32, 32, 32});
	int3 nranks3D{2, 2, 2};
	uDeviceX udevice(argc, argv, nranks3D, globalDomainSize, logger, "poiseuille.log", 3);

	SimulationPlugin*  simPl;
	PostprocessPlugin* postPl;
	if (udevice.isComputeTask())
	{
		Integrator  constDP = createIntegrator(config.child("simulation").child("integrator"));
		Interaction dpdInt = createInteraction(config.child("simulation").child("interaction"));
		InitialConditions dpdIc = createIC(config.child("simulation").child("particle_vector"));
		Wall wall = createWall(config.child("simulation").child("wall"));

		ParticleVector* dpd = new ParticleVector(config.child("simulation").child("particle_vector").attribute("name").as_string());

		udevice.sim->registerParticleVector(dpd, &dpdIc);

		udevice.sim->registerIntegrator(&constDP);
		udevice.sim->registerInteraction(&dpdInt);
		udevice.sim->registerWall(&wall);

		udevice.sim->setIntegrator("dpd", "const_dp");
		udevice.sim->setInteraction("dpd", "dpd", "dpd_int");

		simPl = new SimulationStats("stats", 100);
		//simPl = new Avg3DPlugin("dpd", 10, 2000, {32, 32, 32}, {0.5, 0.5, 0.5}, true, true, true);
	}
	else
	{
		postPl = new PostprocessStats("stats");
		//postPl = new Avg3DDumper("xdmf/avgfields", nranks3D);
	}

	udevice.registerJointPlugins(simPl, postPl);
	udevice.run();

	return 0;
}
