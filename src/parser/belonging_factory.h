//================================================================================================
// Object belonging
//================================================================================================

#pragma once

#include <core/xml/pugixml.hpp>

#include <core/object_belonging/ellipsoid_belonging.h>
#include <core/object_belonging/mesh_belonging.h>

#include <core/utils/make_unique.h>

class ObjectBelongingCheckerFactory
{
private:
	static std::unique_ptr<ObjectBelongingChecker> createMeshBelongingChecker(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");

		return std::make_unique<MeshBelongingChecker>(name);
	}

	static std::unique_ptr<ObjectBelongingChecker> createEllipsoidBelongingChecker(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string("");

		return std::make_unique<EllipsoidBelongingChecker>(name);
	}

public:
	static std::unique_ptr<ObjectBelongingChecker> create(pugi::xml_node node)
	{
		std::string type = node.attribute("type").as_string();

		if (type == "mesh")
			return createMeshBelongingChecker(node);

		if (type == "analytical_ellipsoid")
			return createEllipsoidBelongingChecker(node);

		die("Unable to parse input at %s, unknown 'type': '%s'", node.path().c_str(), type.c_str());

		return nullptr;
	}
};
