#include "parser.h"

#include <core/udevicex.h>
#include <core/simulation.h>
#include <core/postproc.h>

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/pvs/rbc_vector.h>

#include <core/mesh.h>

#include <core/initial_conditions/uniform_ic.h>
#include <core/initial_conditions/ellipsoid_ic.h>
#include <core/initial_conditions/rbcs_ic.h>
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
#include <core/walls/stationary_walls/plane.h>

#include <core/bouncers/from_mesh.h>
#include <core/bouncers/from_ellipsoid.h>
#include <core/object_belonging/ellipsoid_belonging.h>
#include <core/object_belonging/mesh_belonging.h>

#include <plugins/dumpavg.h>
#include <plugins/dumpxyz.h>
#include <plugins/stats.h>
#include <plugins/temperaturize.h>
#include <plugins/dump_obj_position.h>
#include <plugins/impose_velocity.h>

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
		auto name = node.attribute("name").as_string();
		auto mass = node.attribute("mass").as_float(1);

		return (ParticleVector*) new ParticleVector(name, mass);
	}

	static ParticleVector* createRigidEllipsoids(pugi::xml_node node)
	{
		auto name    = node.attribute("name").as_string("");
		auto mass    = node.attribute("mass").as_int(1);

		auto objSize = node.attribute("particles_per_obj").as_int(1);
		auto axes    = node.attribute("axes").as_float3( make_float3(1) );

		return (ParticleVector*) new RigidEllipsoidObjectVector(name, mass, objSize, axes);
	}

	static ParticleVector* createRbcs(pugi::xml_node node)
	{
		auto name      = node.attribute("name").as_string("");
		auto mass      = node.attribute("mass").as_int(1);

		auto objSize   = node.attribute("particles_per_obj").as_int(1);

		auto meshFname = node.attribute("mesh_filename").as_string("rbcmesh.off");

		Mesh mesh(meshFname);

		return (ParticleVector*) new RBCvector(name, mass, objSize, std::move(mesh));
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
		auto density = node.attribute("density").as_float(1.0);
		return (InitialConditions*) new UniformIC(density);
	}

	static InitialConditions* createEllipsoidsIC(pugi::xml_node node)
	{
		auto icfname  = node.attribute("ic_filename"). as_string("ellipsoids.ic");
		auto xyzfname = node.attribute("xyz_filename").as_string("ellipsoid.xyz");

		return (InitialConditions*) new EllipsoidIC(xyzfname, icfname);
	}

	static InitialConditions* createRBCsIC(pugi::xml_node node)
	{
		auto icfname  = node.attribute("ic_filename"). as_string("rbcs.ic");
		auto offfname = node.attribute("mesh_filename").as_string("rbc_mesh.off");

		return (InitialConditions*) new RBC_IC(offfname, icfname);
	}

	static InitialConditions* createRestartIC(pugi::xml_node node)
	{
		auto path = node.attribute("path").as_string("restart/");

		return (InitialConditions*) new RestartIC(path);
	}


public:
	static InitialConditions* create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "uniform")
			return createUniformIC(node);
		if (type == "read_ellipsoids")
			return createEllipsoidsIC(node);
		if (type == "read_rbcs")
			return createRBCsIC(node);
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

		auto force = node.attribute("force").as_float(0);

		std::string dirStr = node.attribute("direction").as_string("x");

		Forcing_PeriodicPoiseuille::Direction dir;
		if (dirStr == "x") dir = Forcing_PeriodicPoiseuille::Direction::x;
		if (dirStr == "y") dir = Forcing_PeriodicPoiseuille::Direction::y;
		if (dirStr == "z") dir = Forcing_PeriodicPoiseuille::Direction::z;

		Forcing_PeriodicPoiseuille forcing(force, dir);

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

	static Interaction* createRBC(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");

		RBCParameters p;
		std::string preset = node.attribute("preset").as_string();

		auto setIfNotEmpty_float = [&node] (float& val, const char* name)
		{
			if (!node.attribute(name).empty())
				val = node.attribute(name).as_float();
		};

		if (preset == "lina")
			p = Lina_parameters;
		else
			error("Unknown predefined parameter set for '%s' interaction: '%s'",
					name, preset.c_str());

		setIfNotEmpty_float(p.x0,         "x0");
		setIfNotEmpty_float(p.p,          "p");
		setIfNotEmpty_float(p.ka,         "ka");
		setIfNotEmpty_float(p.kb,         "kb");
		setIfNotEmpty_float(p.kd,         "kd");
		setIfNotEmpty_float(p.kv,         "kv");
		setIfNotEmpty_float(p.gammaC,     "gammaC");
		setIfNotEmpty_float(p.gammaT,     "gammaT");
		setIfNotEmpty_float(p.kbT,        "kbT");
		setIfNotEmpty_float(p.mpow,       "mpow");
		setIfNotEmpty_float(p.theta,      "theta");
		setIfNotEmpty_float(p.totArea0,   "area");
		setIfNotEmpty_float(p.totVolume0, "volume");

		return (Interaction*) new InteractionRBCMembrane(name, p);
	}

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
		if (type == "rbc")
			return createRBC(node);

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

	static Wall* createPlaneWall(pugi::xml_node node)
	{
		auto name   = node.attribute("name").as_string("");

		auto normal = node.attribute("normal").as_float3( make_float3(1, 0, 0) );
		auto point  = node.attribute("point_through").as_float3( );

		StationaryWall_Plane plane(normalize(normal), point);

		return (Wall*) new SimpleStationaryWall<StationaryWall_Plane>(name, std::move(plane));
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
		if (type == "plane")
			return createPlaneWall(node);
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

class ObjectBelongingCheckerFactory
{
private:
	static ObjectBelongingChecker* createMeshBelongingChecker(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");

		return (ObjectBelongingChecker*) new MeshBelongingChecker(name);
	}

	static ObjectBelongingChecker* createEllipsoidBelongingChecker(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");

		return (ObjectBelongingChecker*) new EllipsoidBelongingChecker(name);
	}

public:
	static ObjectBelongingChecker* create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "mesh")
			return createMeshBelongingChecker(node);

		if (type == "analytical_ellipsoid")
			return createEllipsoidBelongingChecker(node);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return nullptr;
	}
};


//================================================================================================
// Plugins
//================================================================================================

class PluginFactory
{
private:
	static std::pair<SimulationPlugin*, PostprocessPlugin*> createImposeVelocityPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name   = node.attribute("name").as_string();
		auto pvName = node.attribute("pv_name").as_string();

		auto every  = node.attribute("every").as_int(5);
		auto low    = node.attribute("low").as_float3();
		auto high   = node.attribute("high").as_float3();
		auto target = node.attribute("target_velocity").as_float3();

		auto simPl = computeTask ? new ImposeVelocityPlugin(name, pvName, low, high, target, every) : nullptr;

		return { (SimulationPlugin*) simPl, nullptr };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createTemperaturizePlugin(pugi::xml_node node, bool computeTask)
	{
		auto name    = node.attribute("name").as_string();

		auto pvName  = node.attribute("pv_name").as_string();
		auto kbT     = node.attribute("kbt").as_float();
		auto keepVel = node.attribute("keep_velocity").as_bool(false);

		auto simPl = computeTask ? new TemperaturizePlugin(name, pvName, kbT, keepVel) : nullptr;

		return { (SimulationPlugin*) simPl, nullptr };
	}


	static std::pair<SimulationPlugin*, PostprocessPlugin*> createStatsPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name   = node.attribute("name").as_string();

		auto every  = node.attribute("every").as_int(1000);

		auto simPl  = computeTask ? new SimulationStats(name, every) : nullptr;
		auto postPl = computeTask ? nullptr :new PostprocessStats(name);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createDumpavgPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name        = node.attribute("name").as_string();

		auto pvNames     = node.attribute("pv_names").as_string();
		auto sampleEvery = node.attribute("sample_every").as_int(50);
		auto dumpEvery   = node.attribute("dump_every").as_int(5000);
		auto binSize     = node.attribute("bin_size").as_float3( {1, 1, 1} );
		auto momentum    = node.attribute("need_momentum").as_bool(true);
		auto force       = node.attribute("need_force").as_bool(false);

		auto path        = node.attribute("path").as_string("xdmf");

		auto simPl  = computeTask ? new Avg3DPlugin(name, pvNames, sampleEvery, dumpEvery, binSize, momentum, force) : nullptr;
		auto postPl = computeTask ? nullptr : new Avg3DDumper(name, path);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createDumpXYZPlugin(pugi::xml_node node, bool computeTask)
	{
		auto name      = node.attribute("name").as_string();

		auto pvName    = node.attribute("pv_name").as_string();
		auto dumpEvery = node.attribute("dump_every").as_int(1000);

		auto path      = node.attribute("path").as_string("xyz/");

		auto simPl  = computeTask ? new XYZPlugin(name, pvName, dumpEvery) : nullptr;
		auto postPl = computeTask ? nullptr : new XYZDumper(name, path);

		return { (SimulationPlugin*) simPl, (PostprocessPlugin*) postPl };
	}

	static std::pair<SimulationPlugin*, PostprocessPlugin*> createDumpObjPosition(pugi::xml_node node, bool computeTask)
	{
		auto name      = node.attribute("name").as_string();

		auto ovName    = node.attribute("ov_name").as_string();
		auto dumpEvery = node.attribute("dump_every").as_int(1000);

		auto path      = node.attribute("path").as_string("pos/");

		auto simPl  = computeTask ? new ObjPositionsPlugin(name, ovName, dumpEvery) : nullptr;
		auto postPl = computeTask ? nullptr : new ObjPositionsDumper(name, path);

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
		if (type == "dump_obj_pos")
			return createDumpObjPosition(node, computeTask);
		if (type == "impose_velocity")
			return createImposeVelocityPlugin(node, computeTask);

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









