#include <functional>
#include <core/xml/pugixml.hpp>
#include <core/wall.h>

class ParticleVector;
class CellList;

//namespace uDeviceX
//{
	struct Integrator
	{
		std::string name;
		float dt;

		std::function<void(ParticleVector*, const float, cudaStream_t)> integrate;
		void exec (ParticleVector* pv, cudaStream_t stream)
		{
			integrate(pv, dt, stream);
		}
	};

	struct Interaction
	{
		float rc;
		std::string name;

		std::function<void(ParticleVector*, CellList*, const float, cudaStream_t)> self;
		std::function<void(ParticleVector*, ParticleVector*, CellList*, const float, cudaStream_t)> halo;
		std::function<void(ParticleVector*, ParticleVector*, CellList*, const float, cudaStream_t)> external;

		void execSelf (ParticleVector* pv, CellList* cl, const float t, cudaStream_t stream)
		{
			self(pv, cl, t, stream);
		}

		void execHalo (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream)
		{
			halo(pv1, pv2, cl, t, stream);
		}

		void execExternal (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream)
		{
			external(pv1, pv2, cl, t, stream);
		}
	};

	struct InitialConditions
	{
		std::function<void(MPI_Comm&, ParticleVector*, float3, float3)> exec;
	};

	Integrator  createIntegrator(pugi::xml_node node);
	Interaction createInteraction(pugi::xml_node node);
	InitialConditions createIC(pugi::xml_node node);
	Wall createWall(pugi::xml_node node);
//}
