//================================================================================================
// Particle vectors
//================================================================================================

#pragma once

#include <core/xml/pugixml.hpp>

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/pvs/rbc_vector.h>

#include <core/mesh.h>

class ParticleVectorFactory
{
private:
	static ParticleVector* createRegularPV(pugi::xml_node node)
	{
		auto name = node.attribute("name").as_string();
		auto mass = node.attribute("mass").as_float(1.0);

		return (ParticleVector*) new ParticleVector(name, mass);
	}

	static ParticleVector* createRigidEllipsoids(pugi::xml_node node)
	{
		auto name    = node.attribute("name").as_string("");
		auto mass    = node.attribute("mass").as_float(1);

		auto objSize = node.attribute("particles_per_obj").as_int(1);
		auto axes    = node.attribute("axes").as_float3( make_float3(1) );

		return (ParticleVector*) new RigidEllipsoidObjectVector(name, mass, objSize, axes);
	}

	static ParticleVector* createRigidObjects(pugi::xml_node node)
	{
		auto name      = node.attribute("name").as_string("");
		auto mass      = node.attribute("mass").as_float(1);

		auto objSize   = node.attribute("particles_per_obj").as_int(1);
		auto J         = node.attribute("moment_of_inertia").as_float3();
		auto meshFname = node.attribute("mesh_filename").as_string("mesh.off");

		Mesh mesh(meshFname);

		return (ParticleVector*) new RigidObjectVector(name, mass, J, objSize, std::move(mesh));
	}

	static ParticleVector* createRbcs(pugi::xml_node node)
	{
		auto name      = node.attribute("name").as_string("");
		auto mass      = node.attribute("mass").as_float(1);

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
		if (type == "rigid_objects")
			return createRigidObjects(node);
		if (type == "rbcs")
			return createRbcs(node);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());
		return nullptr;
	}
};
