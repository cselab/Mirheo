#include "particle_vectors.h"

#include <core/pvs/particle_vector.h>

PySimpleParticleVector::PySimpleParticleVector(std::string name, float mass) :
    PyParticleVector(new ParticleVector(name, mass))
{    }
