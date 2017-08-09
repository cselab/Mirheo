#pragma once

#include <functional>
#include <core/xml/pugixml.hpp>

class ParticleVector;

struct InitialConditions
{
	virtual void exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 subDomainSize) = 0;

	virtual ~InitialConditions();
};

struct UniformIC : InitialConditions
{
	float mass, density;

	UniformIC(pugi::xml_node node);

	void exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 subDomainSize);

	virtual ~UniformIC() = default;
};
