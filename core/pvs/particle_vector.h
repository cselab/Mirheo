#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/domain.h>

#include "extra_data/extra_data_manager.h"

#include <map>

class LocalParticleVector
{
protected:
	int np;

public:

	PinnedBuffer<Particle> coosvels;
	DeviceBuffer<Force> forces;
	ExtraDataManager extraPerParticle;

	// Local coordinate system; (0,0,0) is center of the local domain
	LocalParticleVector(int n=0, cudaStream_t stream = 0)
	{
		resize(n, stream);
		// usually old positions and velocities don't need to exchanged
		extraPerParticle.createData<Particle> ("old_particles", n);
	}

	int size()
	{
		return np;
	}

	virtual void resize(const int n, cudaStream_t stream)
	{
		if (n < 0) die("Tried to resize PV to %d < 0 particles", n);

		coosvels.        resize(n, stream);
		forces.          resize(n, stream);
		extraPerParticle.resize(n, stream);

		np = n;
	}

	virtual void resize_anew(const int n)
	{
		if (n < 0) die("Tried to resize PV to %d < 0 particles", n);

		coosvels.        resize_anew(n);
		forces.          resize_anew(n);
		extraPerParticle.resize_anew(n);

		np = n;
	}

	virtual ~LocalParticleVector() = default;
};


class ParticleVector
{
public:
	DomainInfo domain;
	LocalParticleVector *_local, *_halo;

	float mass;
	std::string name;
	// Local coordinate system; (0,0,0) is center of the local domain

	bool haloValid = false;
	bool redistValid = false;

	int cellListStamp{0};

protected:
	ParticleVector(	std::string name, float mass, LocalParticleVector *local, LocalParticleVector *halo ) :
		name(name), mass(mass), _local(local), _halo(halo) {}

public:
	ParticleVector(std::string name, float mass, int n=0) :
		name(name), mass(mass),
		_local( new LocalParticleVector(n) ),
		_halo ( new LocalParticleVector(0) )
	{}

	LocalParticleVector* local() { return _local; }
	LocalParticleVector* halo()  { return _halo;  }

	virtual void checkpoint(MPI_Comm comm, std::string path);
	virtual void restart(MPI_Comm comm, std::string path);

	virtual ~ParticleVector() { delete _local; delete _halo; }
};

#include "views/pv.h"



