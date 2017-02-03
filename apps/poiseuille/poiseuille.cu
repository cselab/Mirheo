#include <core/simulation.h>
#include <plugins/plugin.h>
#include <plugins/dumpavg.h>
#include <core/xml/pugixml.hpp>
#include <core/wall.h>

Logger logger;

int main(int argc, char** argv)
{
	pugi::xml_document config;
	pugi::xml_parse_result result = config.load_file("poiseuille.xml");

	float3 globalDomainSize{16, 16, 16};
	int3 nranks3D{1, 1, 1};
	//uDeviceX udevice(argc, argv, nranks3D, fullDomainSize, logger, "poiseuille.log", 9);


	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
		printf("ERROR: The MPI library does not have full thread support\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	logger.init(MPI_COMM_WORLD, "poiseuille.log", 4);

	MPI_Comm comm;
	MPI_Check( MPI_Comm_dup(MPI_COMM_WORLD, &comm) );
	Simulation* sim = new Simulation(nranks3D, globalDomainSize, comm, comm);

	SimulationPlugin*  simPl;
	PostprocessPlugin* postPl;
	//if (udevice.isComputeTask())
	{
		Integrator  constDP = createIntegrator(config.child("simulation").child("integrator"));
		Interaction dpdInt = createInteraction(config.child("simulation").child("interaction"));
		InitialConditions dpdIc = createIC(config.child("simulation").child("particle_vector"));
		Wall wall = createWall(config.child("simulation").child("wall"));


		ParticleVector* dpd = new ParticleVector(config.child("simulation").child("particle_vector").attribute("name").as_string());

		sim->registerParticleVector(dpd, &dpdIc);

		sim->registerIntegrator(&constDP);
		sim->registerInteraction(&dpdInt);
		sim->registerWall(&wall);

		sim->setIntegrator("dpd", "const_dp");
		sim->setInteraction("dpd", "dpd", "dpd_int");

		//simPl = new Avg3DPlugin("dpd", 10, 2000, {32, 32, 32}, {0.5, 0.5, 0.5}, true, true, true);
	}
	//else
	//{
	//	postPl = new Avg3DDumper("xdmf/avgfields", nranks3D);
	//}

	//udevice.registerJointPlugins(simPl, postPl);
	sim->run(1000);

	return 0;
}
