//================================================================================================
// Walls
//================================================================================================

#pragma once

#include <core/xml/pugixml.hpp>

#include <core/walls/simple_stationary_wall.h>

#include <core/walls/stationary_walls/sdf.h>
#include <core/walls/stationary_walls/sphere.h>
#include <core/walls/stationary_walls/cylinder.h>
#include <core/walls/stationary_walls/plane.h>
#include <core/walls/stationary_walls/box.h>

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

	static Wall* createBoxWall(pugi::xml_node node)
	{
		auto name   = node.attribute("name").as_string("");

		auto high   = node.attribute("high").as_float3();
		auto low    = node.attribute("low") .as_float3();
		auto inside = node.attribute("inside").as_bool(false);

		StationaryWall_Box box(high, low, inside);

		return (Wall*) new SimpleStationaryWall<StationaryWall_Box>(name, std::move(box));
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
		if (type == "box")
			return createSphereWall(node);
		if (type == "plane")
			return createPlaneWall(node);
		if (type == "sdf")
			return createSDFWall(node);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return nullptr;
	}
};

