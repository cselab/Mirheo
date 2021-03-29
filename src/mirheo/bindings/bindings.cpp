// Copyright 2020 ETH Zurich. All Rights Reserved.
#include <mirheo/core/logger.h>
#include <mirheo/core/version.h>
#include "bindings.h"
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
    exportLocalParticleVector(pv);
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
