//================================================================================================
// Integrators
//================================================================================================

#pragma once

#include <core/xml/pugixml.hpp>

#include <core/integrators/vv.h>
#include <core/integrators/const_omega.h>
#include <core/integrators/oscillate.h>
#include <core/integrators/rigid_vv.h>

#include <core/integrators/forcing_terms/none.h>
#include <core/integrators/forcing_terms/const_dp.h>
#include <core/integrators/forcing_terms/periodic_poiseuille.h>

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

	static Integrator* createOscillating(pugi::xml_node node)
	{
		auto name    = node.attribute("name").as_string();
		auto dt      = node.attribute("dt").as_float(0.01);

		auto vel     = node.attribute("velocity").as_float3();
		auto period  = node.attribute("period").as_int();

		return (Integrator*) new IntegratorOscillate(name, dt, vel, period);
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
		if (type == "oscillate")
			return createOscillating(node);
		if (type == "rigid_vv")
			return createRigidVV(node);

		die("Unable to parse input at %s, unknown 'type' %s", node.path().c_str(), type.c_str());

		return nullptr;
	}
};
