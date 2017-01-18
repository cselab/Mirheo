#include "../core/simulation.h"
#include "../plugins/plugin.h"
#include "../plugins/dumpavg.h"
#include "../core/iniparser.h"


int main(int argc, char** argv)
{
	uDeviceX udevice;

	float3 fullDomainSize{16, 16, 16};
	udevice.sim = new Simulation(fullDomainSize);
	udevice.post = new Postprocess();

	Integrator dpdIntegrator(IntegratorType::constDP,
		cfgFile.getFloat("common", "dt"),
		cfgFile.getFloat3("dpd_integrator", "force"));

	Interaction dpdInteraction(InteractionType::dpd,
			1.0f,
			cfgFile.getFloat("dpd", "aij"),
			cfgFile.getFloat("dpd", "sigma"),
			cfgFile.getFloat("dpd", "gamma"),
			cfgFile.getFloat("common", "dt"));

	Wall wall("wall", "tube.sdf");

	udevice.sim->registerPV("dpd", PVType::particles, dpdIntegrator);
	udevice.sim->registerInteraction("dpd", "dpd", dpdInteraction);
	udevice.sim->registerWall("wall", wall);

	Avg3DPlugin avg(1, udevice->sim, comm, rank, "dpd", 10, 2000, {32, 32, 32}, {0.5, 0.5, 0.5}, true, true, true);
	Avg3DDumper dump(1, comm, rnk, "xdmf/avgfields");

	udevice.sim->registerPlugin(&avg);
	udevice.post->registerPlugin(&dump);

	udevice.run();

	return 0;
}
