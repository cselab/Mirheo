//================================================================================================
// Initial conditions
//================================================================================================

#pragma once

#include <core/xml/pugixml.hpp>

#include <core/initial_conditions/uniform_ic.h>
#include <core/initial_conditions/rigid_ic.h>
#include <core/initial_conditions/rbcs_ic.h>
#include <core/initial_conditions/restart.h>

class InitialConditionsFactory
{
private:
	static InitialConditions* createUniformIC(pugi::xml_node node)
	{
		auto density = node.attribute("density").as_float(1.0);
		return (InitialConditions*) new UniformIC(density);
	}

	static InitialConditions* createRigidIC(pugi::xml_node node)
	{
		auto icfname  = node.attribute("ic_filename"). as_string("objects.ic");
		auto xyzfname = node.attribute("xyz_filename").as_string("object.xyz");

		return (InitialConditions*) new RigidIC(xyzfname, icfname);
	}

	static InitialConditions* createRBCsIC(pugi::xml_node node)
	{
		auto icfname  = node.attribute("ic_filename"). as_string("rbcs.ic");

		return (InitialConditions*) new RBC_IC(icfname);
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
		if (type == "read_rigid")
			return createRigidIC(node);
		if (type == "read_rbcs")
			return createRBCsIC(node);
		if (type == "restart")
			return createRestartIC(node);


		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return nullptr;
	}
};
