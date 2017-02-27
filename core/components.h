#include <functional>
#include <core/xml/pugixml.hpp>
#include <core/wall.h>
#include <core/interactions.h>

class ParticleVector;
class CellList;

//namespace uDeviceX
//{
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

		void exec (InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream)
		{
			interaction(type, pv1, pv2, cl, t, stream);
		}
	};

	struct InitialConditions
	{
		std::function<void(const MPI_Comm&, ParticleVector*, float3, float3)> exec;
	};

	Integrator  createIntegrator(pugi::xml_node node);
	Interaction createInteraction(pugi::xml_node node);
	InitialConditions createIC(pugi::xml_node node);
	Wall createWall(pugi::xml_node node);
//}
