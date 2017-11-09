//================================================================================================
// Bouncers
//================================================================================================

#pragma once

#include <core/xml/pugixml.hpp>

#include <core/bouncers/from_mesh.h>
#include <core/bouncers/from_ellipsoid.h>

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
