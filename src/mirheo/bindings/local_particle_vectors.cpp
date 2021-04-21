// Copyright 2021 ETH Zurich. All Rights Reserved.

#include "local_particle_vectors.h"
#include "cuda_array_interface.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/object_vector.h>

namespace mirheo
{

using namespace py::literals;

void exportLocalParticleVectors(py::module& m)
{
    py::class_<LocalParticleVector> pylpv(m, "LocalParticleVector", R"(
        Particle local data storage, composed of particle channels.
    )");

    pylpv.def_readonly("per_particle", &LocalParticleVector::dataPerParticle, R"(
            The :any:`DataManager` that contains the particle channels.
        )", py::return_value_policy::reference_internal);



    py::class_<LocalObjectVector> pylov(m, "LocalObjectVector", pylpv, R"(
        Object vector local data storage, additionally contains object channels.
    )");

    pylov.def_readonly("per_object", &LocalObjectVector::dataPerObject, R"(
            The :any:`DataManager` that contains the object channels.
        )", py::return_value_policy::reference_internal);

}

} // namespace mirheo
