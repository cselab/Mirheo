#include <pybind11/pybind11.h>
#include <core/logger.h>
#include "bindings.h"

namespace py = pybind11;

Logger logger;

PYBIND11_MODULE(_udevicex, m)
{
    exportUdevicex(m);
    
    auto ic = m.def_submodule("InitialConditions");
    exportInitialConditions(ic);

    auto pv = m.def_submodule("ParticleVectors");
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
    
    auto plugins = m.def_submodule("Plugins");
    exportPlugins(plugins);
}
