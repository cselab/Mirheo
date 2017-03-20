#include <core/simulation.h>
#include <plugins/plugin.h>
#include <plugins/stats.h>
#include <plugins/dumpavg.h>
#include <core/xml/pugixml.hpp>
#include <core/wall.h>

Logger logger;

int main(int argc, char** argv)
{
	pugi::xml_document config;
	pugi::xml_parse_result result = config.load_file("poiseuille.xml");

	float3 globalDomainSize = config.child("simulation").child("domain").attribute("size").as_float3({32, 32, 32});
	int3 nranks3D = config.child("simulation").attribute("mpi_ranks").as_int3({1, 1, 1});
	uDeviceX udevice(argc, argv, nranks3D, globalDomainSize, logger, "poiseuille.log",
			config.child("simulation").attribute("debug_lvl").as_int(5), config.child("simulation").attribute("debug_lvl").as_int(5) >= 10);

	SimulationPlugin  *simStat,  *simAvg;
	PostprocessPlugin *postStat, *postAvg;
	if (udevice.isComputeTask())
	{
		Integrator  constDP = createIntegrator(config.child("simulation").child("integrator"));
		Interaction dpdInt = createInteraction(config.child("simulation").child("interaction"));

		InitialConditions dpdIc  = createIC(config.child("simulation").child("particle_vector"));
		InitialConditions dpdIc2 = createIC(config.child("simulation").child("particle_vector").next_sibling());

		Wall wall = createWall(config.child("simulation").child("wall"));

		ParticleVector* dpd  = new ParticleVector(config.child("simulation").child("particle_vector").attribute("name").as_string());
		//ParticleVector* dpd2 = new ParticleVector(config.child("simulation").child("particle_vector").next_sibling().attribute("name").as_string());

		udevice.sim->registerParticleVector(dpd, &dpdIc);
		//udevice.sim->registerParticleVector(dpd2, &dpdIc2);

		udevice.sim->registerIntegrator(&constDP);
		udevice.sim->registerInteraction(&dpdInt);
		udevice.sim->registerWall(&wall);

		udevice.sim->setIntegrator("dpd", "const_dp");
		//udevice.sim->setIntegrator("dpd2", "const_dp");
		udevice.sim->setInteraction("dpd", "dpd", "dpd_int");
		//udevice.sim->setInteraction("dpd2", "dpd", "dpd_int");
		//udevice.sim->setInteraction("dpd2", "dpd2", "dpd_int");
		udevice.sim->setInteraction("dpd", "wall", "dpd_int");

		simStat = new SimulationStats("stats", 500);
		simAvg  = new Avg3DPlugin("averaging", "dpd", 10, 5000, {1, 1, 1}, true, true);
	}
	else
	{
		postStat = new PostprocessStats("stats");
		postAvg = new Avg3DDumper("averaging", "xdmf/avgfields", nranks3D);
	}

	udevice.registerJointPlugins(simStat, postStat);
	udevice.registerJointPlugins(simAvg,  postAvg);

	const int niters = config.child("simulation").child("run").attribute("stop_time").as_float(10.0f) /
					   config.child("simulation").child("run").attribute("dt")       .as_float(0.01f);
	udevice.run(niters);

	// ???
	MPI_Finalize();

	return 0;
}
