// Copyright 2020 ETH Zurich. All Rights Reserved.

#include "bouncers.h"
#include "cuda_array_interface.h"
#include "data_manager.h"
#include "initial_conditions.h"
#include "integrators.h"
#include "interactions.h"
#include "local_particle_vectors.h"
#include "mirheo.h"
#include "object_belonging_checkers.h"
#include "particle_vectors.h"
#include "plugins.h"
#include "utils.h"
#include "vector_types.h"
#include "walls.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/version.h>

#include <mpi.h>

PYBIND11_MODULE(libmirheo, m)
{
    using namespace mirheo;

    // Is there an equivalent of cls.def_readonly_static(...) for modules?
    // https://github.com/pybind/pybind11/issues/1869
    m.attr("version") = Version::mir_version;  // This is not const!

    exportCudaArrayInterface(m);
    exportVectorTypes(m);
    exportConfigValue(m);
    exportUnitConversion(m);

    exportMirheo(m);

    auto ic = m.def_submodule("InitialConditions");
    exportInitialConditions(ic);

    auto pv = m.def_submodule("ParticleVectors");
    exportDataManager(pv);
    exportLocalParticleVectors(pv);
    exportParticleVectors(pv);

    auto interactions = m.def_submodule("Interactions");
    exportInteractions(interactions);

    auto integrators = m.def_submodule("Integrators");
    exportIntegrators(integrators);

    auto checkers = m.def_submodule("BelongingCheckers");
    exportObjectBelongingCheckers(checkers);

    auto bouncers = m.def_submodule("Bouncers");
    exportBouncers(bouncers);

    auto walls = m.def_submodule("Walls");
    exportWalls(walls);

    auto utils = m.def_submodule("Utils");
    exportUtils(utils);

    auto plugins = m.def_submodule("Plugins");
    exportPlugins(plugins);

    m.def("destroyCudaContext", [] () { cudaDeviceReset(); });
    m.def("abort", [] () { MPI_Abort(MPI_COMM_WORLD, -1); }, "Abort the program and quit all the MPI processes");
}
