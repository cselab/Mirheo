#pragma once

#include <functional>
#include <string>
#include <core/xml/pugixml.hpp>


class ParticleVector;
class RigidObjectVector;

struct Integrator
{
	std::string name;
	float dt;

	virtual void stage1(ParticleVector* pv, cudaStream_t stream) = 0;
	virtual void stage2(ParticleVector* pv, cudaStream_t stream) = 0;

	Integrator(pugi::xml_node node) :
		dt(   node.attribute("dt").  as_float(0.01) ),
		name( node.attribute("name").as_string("")  )
	{}

	virtual ~Integrator() = default;
};


struct IntegratorVVNoFlow : Integrator
{
	void stage1(ParticleVector* pv, cudaStream_t stream);
	void stage2(ParticleVector* pv, cudaStream_t stream);

	IntegratorVVNoFlow(pugi::xml_node node) :
		Integrator(node)
	{}

	~IntegratorVVNoFlow() = default;
};

struct IntegratorVVConstDP : Integrator
{
	float3 extraForce;

	void stage1(ParticleVector* pv, cudaStream_t stream);
	void stage2(ParticleVector* pv, cudaStream_t stream);

	IntegratorVVConstDP(pugi::xml_node node) :
		Integrator(node),
		extraForce( node.attribute("extra_force").as_float3({0,0,0}) )
	{}

	~IntegratorVVConstDP() = default;
};

struct IntegratorConstOmega : Integrator
{
	float3 center, omega;

	void stage1(ParticleVector* pv, cudaStream_t stream) {};
	void stage2(ParticleVector* pv, cudaStream_t stream);

	IntegratorConstOmega(pugi::xml_node node) :
		Integrator(node),
		center( node.attribute("center").as_float3({0,0,0}) ),
		omega ( node.attribute("omega"). as_float3({0,0,0}) )
	{}

	~IntegratorConstOmega() = default;
};

struct IntegratorVVRigid : Integrator
{
	void stage1(ParticleVector* pv, cudaStream_t stream) {};
	void stage2(ParticleVector* pv, cudaStream_t stream);

	IntegratorVVRigid(pugi::xml_node node) :
		Integrator(node)
	{}

	~IntegratorVVRigid() = default;
};
