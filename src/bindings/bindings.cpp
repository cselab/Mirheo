#include <extern/pybind11/include/pybind11/pybind11.h>
#include <core/logger.h>
#include "bindings.h"

namespace py = pybind11;

Logger logger;

PYBIND11_MODULE(_udevicex, m)
{
    exportUdevicex(m);
    exportInitialConditions(m);
    exportParticleVectors(m);
    exportIntegrators(m);
    exportInteractions(m);
}
