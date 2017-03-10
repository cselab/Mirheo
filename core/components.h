#include <functional>
#include <core/xml/pugixml.hpp>
#include <core/wall.h>
#include <core/interactions.h>
#include <core/particle_vector.h>

class ParticleVector;
class CellList;

struct Integrator
{
	std::string name;
	float dt;

	std::function<void(ParticleVector*, const float, cudaStream_t)> integrator;
	void exec (ParticleVector* pv, cudaStream_t stream)
	{
		integrator(pv, dt, stream);
	}
};

struct Interaction
{
	float rc;
	std::string name;

	std::function<void(InteractionType, ParticleVector*, ParticleVector*, CellList*, const float, cudaStream_t)> interaction;

	void exec (InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
	{
		// This should ideally be moved to interactions.cu,
		// but because of CUDA lambda limitations it would
		// swell the code by a looooot
		if (type == InteractionType::Regular)
		{
			if (pv1->np < pv2->np)
				interaction(type, pv1, pv2, cl1, t, stream);
			else
				interaction(type, pv2, pv1, cl2, t, stream);
		}

		if (type == InteractionType::Halo)
		{
			interaction(type, pv1, pv2, cl1, t, stream);

			if(pv1 != pv2)
				interaction(type, pv2, pv1, cl2, t, stream);
		}
	}
};

struct InitialConditions
{
	std::function<void(const MPI_Comm&, ParticleVector*, float3, float3)> exec;
};

Integrator   createIntegrator(pugi::xml_node node);
Interaction createInteraction(pugi::xml_node node);
InitialConditions    createIC(pugi::xml_node node);
Wall               createWall(pugi::xml_node node);
