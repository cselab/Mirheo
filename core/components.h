#include <functional>

#include "containers.h"
#include "xml/pugixml.hpp"

//namespace uDeviceX
//{
	struct Integrator
	{
		std::string name;

		std::function<void(ParticleVector*, const float, cudaStream_t)> integrate;
		void exec (ParticleVector* pv, const float dt, cudaStream_t stream)
		{
			integrate(pv, dt, stream);
		}
	};

	struct Interaction
	{
		float rc;
		std::string name;

		std::function<void(ParticleVector*, CellList*, const float, cudaStream_t)> self;
		std::function<void(ParticleVector*, CellList*, const float, cudaStream_t)> halo;
		std::function<void(ParticleVector*, ParticleVector*, CellList*, const float, cudaStream_t)> external;

		void execSelf (ParticleVector* pv, CellList* cl, const float time, cudaStream_t stream)
		{
			self(pv, cl, time, stream);
		}

		void execHalo (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float time, cudaStream_t stream)
		{
			external(pv1, pv2, cl, time, stream);
		}

		void execExternal (ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float time, cudaStream_t stream)
		{
			external(pv1, pv2, cl, time, stream);
		}
	};

	Integrator  createIntegrator(std::string name, pugi::xml_node node);
	Interaction createInteraction(std::string name, pugi::xml_node node);
//}
