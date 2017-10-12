#include "parser.h"

#include <core/udevicex.h>
#include <core/simulation.h>

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/pvs/rbc_vector.h>

#include <core/initial_conditions/uniform.h>
#include <core/initial_conditions/ellipsoid.h>
#include <core/initial_conditions/restart.h>

#include <core/integrators/vv.h>
#include <core/integrators/const_omega.h>
#include <core/integrators/rigid_vv.h>

#include <core/integrators/forcing_terms/none.h>
#include <core/integrators/forcing_terms/const_dp.h>
#include <core/integrators/forcing_terms/periodic_poiseuille.h>

#include <core/interactions/pairwise.h>
#include <core/interactions/sampler.h>
#include <core/interactions/rbc.h>

#include <core/interactions/pairwise_interactions/dpd.h>
#include <core/interactions/pairwise_interactions/lj.h>
#include <core/interactions/pairwise_interactions/lj_object_aware.h>

#include <core/walls/simple_stationary_wall.h>

#include <core/walls/stationary_walls/sdf.h>
#include <core/walls/stationary_walls/sphere.h>
#include <core/walls/stationary_walls/cylinder.h>

#include <core/bouncers/from_mesh.h>
#include <core/bouncers/from_ellipsoid.h>

#include <plugins/dumpavg.h>
#include <plugins/dumpxyz.h>
#include <plugins/stats.h>
#include <plugins/temperaturize.h>

#include <core/xml/pugixml.hpp>

#include <utility>

//================================================================================================
// Particle vectors
//================================================================================================

class ParticleVectorFactory
{
private:
	static ParticleVector* createRegularPV(pugi::xml_node node)
	{
		std::string name = node.attribute("name").as_string();
		float mass = node.attribute("mass").as_float(1);

		return (ParticleVector*) new ParticleVector(name, mass);
	}

	static ParticleVector* createRigidEllipsoids(pugi::xml_node node)
	{
		std::string name = node.attribute("name").as_string("");
		float mass  = node.attribute("mass").as_int(1);
		int objSize = node.attribute("particles_per_obj").as_int(1);
		float3 axes = node.attribute("axes").as_float3( make_float3(1) );

		return (ParticleVector*) new RigidEllipsoidObjectVector(name, mass, objSize, axes);
	}

	static ParticleVector* createRbcs(pugi::xml_node node)
	{
		std::string name = node.attribute("name").as_string("");
		float mass  = node.attribute("mass").as_int(1);
		int objSize = node.attribute("particles_per_obj").as_int(1);

		std::string meshFname = node.attribute("mesh_filename").as_string("rbcmesh.topo");
		ObjectMesh mesh;// = readMeshTopology(meshFname);

		return (ParticleVector*) new RBCvector(name, mass, objSize);//, mesh);
	}

public:
	static ParticleVector* create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "regular")
			return createRegularPV(node);
		if (type == "rigid_ellipsoids")
			return createRigidEllipsoids(node);
		if (type == "rbcs")
			return createRbcs(node);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());
		return nullptr;
	}
};

//================================================================================================
// Initial conditions
//================================================================================================

class InitialConditionsFactory
{
private:
	static InitialConditions* createUniformIC(pugi::xml_node node)
	{
		float density = node.attribute("density").as_float(1.0);
		return (InitialConditions*) new UniformIC(density);
	}

	static InitialConditions* createEllipsoidIC(pugi::xml_node node)
	{
		std::string icfname  = node.attribute("ic_filename").as_string("ellipsoids.ic");
		std::string xyzfname = node.attribute("xyz_filename").as_string("ellipsoid.xyz");

		return (InitialConditions*) new EllipsoidIC(xyzfname, icfname);
	}

	static InitialConditions* createRestartIC(pugi::xml_node node)
	{
		std::string path = node.attribute("path").as_string("restart/");

		return (InitialConditions*) new RestartIC(path);
	}


public:
	static InitialConditions* create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "uniform")
			return createUniformIC(node);
		if (type == "read_ellipsoids")
			return createEllipsoidIC(node);
		if (type == "restart")
			return createRestartIC(node);


		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return nullptr;
	}
};

//================================================================================================
// Integrators
//================================================================================================

class IntegratorFactory
{
private:
	static Integrator* createVV(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string();
		auto dt   = node.attribute("dt").as_float(0.01);

		Forcing_None forcing;

		return (Integrator*) new IntegratorVV<Forcing_None>(name, dt, forcing);
	}

	static Integrator* createVV_constDP(pugi::xml_node node)
	{
		auto name       = node.attribute("name").as_string();
		auto dt         = node.attribute("dt").as_float(0.01);

		auto extraForce = node.attribute("extra_force").as_float3();

		Forcing_ConstDP forcing(extraForce);

		return (Integrator*) new IntegratorVV<Forcing_ConstDP>(name, dt, forcing);
	}

	static Integrator* createVV_PeriodicPoiseuille(pugi::xml_node node)
	{
		auto name  = node.attribute("name").as_string();
		auto dt    = node.attribute("dt").as_float(0.01);

		auto force = node.attribute("force").as_float();

		std::string dirStr = node.attribute("direction").as_string("x");

		// TODO: make all see simulation instead
		float3 domain = node.root().child("simulation").child("domain").attribute("size").as_float3();

		Forcing_PeriodicPoiseuille::Direction dir;
		if (dirStr == "x") dir = Forcing_PeriodicPoiseuille::Direction::x;
		if (dirStr == "y") dir = Forcing_PeriodicPoiseuille::Direction::y;
		if (dirStr == "z") dir = Forcing_PeriodicPoiseuille::Direction::z;

		Forcing_PeriodicPoiseuille forcing(force, dir, domain);

		return (Integrator*) new IntegratorVV<Forcing_PeriodicPoiseuille>(name, dt, forcing);
	}
	static Integrator* createConstOmega(pugi::xml_node node)
	{
		auto name   = node.attribute("name").as_string();
		auto dt     = node.attribute("dt").as_float(0.01);

		auto center = node.attribute("center").as_float3();
		auto omega  = node.attribute("omega") .as_float3();

		return (Integrator*) new IntegratorConstOmega(name, dt, center, omega);
	}

	static Integrator* createRigidVV(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string();
		auto dt   = node.attribute("dt").as_float(0.01);

		return (Integrator*) new IntegratorVVRigid(name, dt);
	}

public:
	static Integrator* create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "vv")
			return createVV(node);
		if (type == "vv_const_dp")
			return createVV_constDP(node);
		if (type == "vv_periodic_poiseuille")
			return createVV_PeriodicPoiseuille(node);
		if (type == "const_omega")
			return createConstOmega(node);
		if (type == "rigid_vv")
			return createRigidVV(node);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return nullptr;
	}
};

//================================================================================================
// Interactions
//================================================================================================

class InteractionFactory
{
private:
	static Interaction* createDPD(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");
		auto rc   = node.attribute("rc").as_float(1.0f);

		auto a     = node.attribute("a")    .as_float(50);
		auto gamma = node.attribute("gamma").as_float(20);
		auto kbT   = node.attribute("kbt")  .as_float(1.0);
		auto dt    = node.attribute("dt")   .as_float(0.01);
		auto power = node.attribute("power").as_float(1.0f);

		Pairwise_DPD dpd(rc, a, gamma, kbT, dt, power);

		return (Interaction*) new InteractionPair<Pairwise_DPD>(name, rc, dpd);
	}

	static Interaction* createLJ(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");
		auto rc   = node.attribute("rc").as_float(1.0f);

		auto epsilon = node.attribute("epsilon").as_float(10.0f);
		auto sigma   = node.attribute("sigma")  .as_float(0.5f);

		Pairwise_LJ lj(rc, sigma, epsilon);

		return (Interaction*) new InteractionPair<Pairwise_LJ>(name, rc, lj);
	}

	static Interaction* createLJ_objectAware(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");
		auto rc   = node.attribute("rc").as_float(1.0f);

		auto epsilon = node.attribute("epsilon").as_float(10.0f);
		auto sigma   = node.attribute("sigma")  .as_float(0.5f);

		Pairwise_LJObjectAware ljo(rc, sigma, epsilon);

		return (Interaction*) new InteractionPair<Pairwise_LJObjectAware>(name, rc, ljo);
	}

//	static Interaction* createMCMCSampler(pugi::xml_node node)
//	{
//		auto name = node.attribute("name").as_string("");
//		auto rc   = node.attribute("rc").as_float(1.0f);
//
//		auto a     = node.attribute("a")    .as_float(50);
//		auto kbT   = node.attribute("kbt")  .as_float(1.0);
//		auto power = node.attribute("power").as_float(1.0f);
//
//		return (Interaction*) new MCMCSampler(name, rc, a, kbT, power);
//	}

public:
	static Interaction* create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "dpd")
			return createDPD(node);
		if (type == "lj")
			return createLJ(node);
		if (type == "lj_object")
			return createLJ_objectAware(node);
//		if (type == "sampler")
//			return createMCMCSampler(node);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return nullptr;
	}
};

//================================================================================================
// Walls
//================================================================================================

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

//================================================================================================
// Bouncers
//================================================================================================

class BouncerFactory
{
private:
	static Bouncer* createMeshBouncer(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");

		return (Bouncer*) new BounceFromMesh(name);
	}

	static Bouncer* createEllipsoidBouncer(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");

		return (Bouncer*) new BounceFromRigidEllipsoid(name);
	}

public:
	static Bouncer* create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "from_mesh")
			return createMeshBouncer(node);

		if (type == "from_ellipsoids")
			return createEllipsoidBouncer(node);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return nullptr;
	}
};

//================================================================================================
// Bouncers
//================================================================================================

class PluginFactory
{
private:
	static std::pair<SimulationPlugin*, PostprocessPlugin*> createTemperaturizePlugin(pugi::xml_node node, bool computeTask)
	{
		std::string name    = node.attribute("name").as_string();
		std::string pvNames = node.attribute("pv_names").as_string();
		float kbT           = node.attribute("kbt").as_float();

		auto simPl = computeTask ? new TemperaturizePlugin(name, pvNames, kbT) : nullptr;

		return { (SimulationPlugin*) simPl, nullptr };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createStatsPlugin(pugi::xml_node node, bool computeTask)
	{
		std::string name = node.attribute("name").as_string();
		int fetchEvery   = node.attribute("every").as_int(1000);

		auto simPl  = computeTask ? new SimulationStats(name, fetchEvery) : nullptr;
		auto postPl = computeTask ? nullptr :new PostprocessStats(name);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createDumpavgPlugin(pugi::xml_node node, bool computeTask)
	{
		std::string name    = node.attribute("name").as_string();
		std::string pvNames = node.attribute("pv_names").as_string();
		int sampleEvery     = node.attribute("sample_every").as_int(50);
		int dumpEvery       = node.attribute("dump_every").as_int(5000);
		float3 binSize      = node.attribute("bin_size").as_float3( {1, 1, 1} );
		bool momentum       = node.attribute("need_momentum").as_bool(true);
		bool force          = node.attribute("need_force").as_bool(false);

		std::string path    = node.attribute("path").as_string("xdmf");

		auto simPl  = computeTask ? new Avg3DPlugin(name, pvNames, sampleEvery, dumpEvery, binSize, momentum, force) : nullptr;
		auto postPl = computeTask ? nullptr : new Avg3DDumper(name, path);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createDumpXYZPlugin(pugi::xml_node node, bool computeTask)
	{
		std::string name   = node.attribute("name").as_string();
		std::string pvName = node.attribute("pv_name").as_string();
		int dumpEvery      = node.attribute("dump_every").as_int(5000);

		std::string path   = node.attribute("path").as_string( ("xyz/" + name).c_str() );

		auto simPl  = computeTask ? new XYZPlugin(name, pvName, dumpEvery) : nullptr;
		auto postPl = computeTask ? nullptr : new XYZDumper(name, path);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}


public:
	static std::pair<SimulationPlugin*, PostprocessPlugin*> create(pugi::xml_node node, bool computeTask)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "temperaturize")
			return createTemperaturizePlugin(node, computeTask);
		if (type == "stats")
			return createStatsPlugin(node, computeTask);
		if (type == "dump_avg_flow")
			return createDumpavgPlugin(node, computeTask);
		if (type == "dump_xyz")
			return createDumpXYZPlugin(node, computeTask);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return {nullptr, nullptr};
	}
};

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

	int3 nranks3D  = simNode.attribute("mpi_ranks").as_int3({1, 1, 1});
	bool noplugins = simNode.attribute("noplugins").as_bool(false);
	int debugLvl   = simNode.attribute("debug_lvl").as_int(5);

	uDeviceX* udx = new uDeviceX(nranks3D, globalDomainSize, logger, logname, debugLvl, noplugins);

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
				udx->sim->registerWall(wall);

				for (auto apply_to : node.children("apply_to"))
					udx->sim->setWallBounce(name,
							apply_to.attribute("pv").as_string(),
							apply_to.attribute("check").as_bool(false));
			}

			if ( std::string(node.name()) == "bounce" )
			{
				auto bouncer = BouncerFactory::create(node);
				auto name = bouncer->name;
				udx->sim->registerBouncer(bouncer);

				for (auto apply_to : node.children("apply_to"))
					udx->sim->setBouncer(name,
							apply_to.attribute("ov").as_string(),
							apply_to.attribute("pv").as_string());
			}
		}
	}

	for (auto node : simNode.children())
		if ( std::string(node.name()) == "plugin" )
		{
			auto simPl_postPl = PluginFactory::create(node, udx->isComputeTask());
			auto simPl  = simPl_postPl.first;
			auto postPl = simPl_postPl.second;

			if (udx->isComputeTask())
			{
				if (simPl->requirePostproc)
					udx->registerJointPlugins(simPl, postPl);
				else
					udx->sim->registerPlugin(simPl);
			}
			else
			{
				if (postPl != nullptr)
					udx->registerJointPlugins(simPl, postPl);
			}
		}

	return udx;
}









