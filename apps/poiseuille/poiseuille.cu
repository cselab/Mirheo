#include <core/simulation.h>
#include <core/interactions/dpd.h>

#include <plugins/plugin.h>
#include <plugins/stats.h>
#include <plugins/dumpavg.h>
#include <plugins/temperaturize.h>

Logger logger;

int main(int argc, char** argv)
{
	pugi::xml_document config;
	pugi::xml_parse_result result = config.load_file("poiseuille.xml");

	float3 globalDomainSize = config.child("simulation").child("domain").attribute("size").as_float3({32, 32, 32});
	int3 nranks3D = config.child("simulation").attribute("mpi_ranks").as_int3({1, 1, 1});
	bool noplugins = config.child("simulation").attribute("noplugins").as_bool(false);

	uDeviceX udevice(argc, argv, nranks3D, globalDomainSize, logger, "poiseuille.log",
			config.child("simulation").attribute("debug_lvl").as_int(5), noplugins);

	SimulationPlugin  *simStat,  *simAvg;
	PostprocessPlugin *postStat, *postAvg;
	if (udevice.isComputeTask())
	{
		// PVs
		ParticleVector* dpd  = new ParticleVector(config.child("simulation").child("particle_vector").attribute("name").as_string());

		InitialConditions* dpdIc  = new UniformIC(config.child("simulation").child("particle_vector"));
		udevice.sim->registerParticleVector(dpd, dpdIc);

		Wall* wall = new Wall(
				config.child("simulation").child("wall").attribute("name").as_string(),
				config.child("simulation").child("wall").attribute("file_name").as_string(),
				config.child("simulation").child("wall").attribute("h").as_float3({0.25, 0.25, 0.25}));

		//udevice.sim->registerWall( wall, true );


		// Manipulators
		Integrator*  constDP = new IntegratorVVConstDP(config.child("simulation").child("integrator"));
		Interaction* dpdInt = new InteractionDPD(config.child("simulation").child("interaction"));

		udevice.sim->registerIntegrator(constDP);
		udevice.sim->registerInteraction(dpdInt);

		udevice.sim->setIntegrator("dpd", "const_dp");
		udevice.sim->setInteraction("dpd", "dpd", "dpd_int");
		//udevice.sim->setInteraction("dpd", "wall", "dpd_int");

		//SimulationPlugin* temp = new TemperaturizePlugin("temp", {"wall"}, 1.0);
		//udevice.sim->registerPlugin(temp);

		simStat = new SimulationStats("stats", 300);
		simAvg  = new Avg3DPlugin("averaging", "dpd", 10, 1000, {1, 1, 1}, true, true);
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
