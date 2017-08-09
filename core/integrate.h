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

	virtual ~Integrator();
};


struct IntegratorVVNoFlow : Integrator
{
	void stage1(ParticleVector* pv, cudaStream_t stream);
	void stage2(ParticleVector* pv, cudaStream_t stream);

	IntegratorVVNoFlow(pugi::xml_node node) {};
};

struct IntegratorVVConstDP : Integrator
{
	float3 extraForce;

	void stage1(ParticleVector* pv, cudaStream_t stream);
	void stage2(ParticleVector* pv, cudaStream_t stream);

	IntegratorVVConstDP(pugi::xml_node node) :
		extraForce( node.attribute("extra_force").as_float3({0,0,0}) )
	{}
};

struct IntegratorConstOmega : Integrator
{
	float3 center, omega;

	void stage1(ParticleVector* pv, cudaStream_t stream);
	void stage2(ParticleVector* pv, cudaStream_t stream);

	IntegratorConstOmega(pugi::xml_node node) :
		center( node.attribute("center").as_float3({0,0,0}) ),
		omega ( node.attribute("omega"). as_float3({0,0,0}) )
	{}
};

struct IntegratorVVRigid : Integrator
{
	void stage1(ParticleVector* pv, cudaStream_t stream);
	void stage2(ParticleVector* pv, cudaStream_t stream);

	IntegratorVVRigid(pugi::xml_node node) {};
};
