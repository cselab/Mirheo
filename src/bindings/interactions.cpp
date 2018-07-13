#include <extern/pybind11/include/pybind11/pybind11.h>

#include <core/pvs/particle_vector.h>

#include <core/interactions/interface.h>
#include <core/interactions/dpd.h>
#include <core/interactions/lj.h>
#include <core/interactions/membrane.h>

namespace py = pybind11;
using namespace pybind11::literals;

void exportInteractions(py::module& m)
{
    // Initial Conditions
    py::class_<Interaction> pyint(m, "Interaction", "hello");

    py::class_<InteractionDPD>(m, "DPD", pyint)
        .def(py::init<std::string, float, float, float, float, float, float>(),
             "name"_a, "rc"_a, "a"_a, "gamma"_a, "kbt"_a, "dt"_a, "power"_a)
        .def("setSpecificPair", &InteractionDPD::setSpecificPair, 
            "pv1"_a, "pv2"_a, "a"_a, "gamma"_a, "kbt"_a, "dt"_a, "power"_a);
        
    py::class_<InteractionLJ>(m, "LJ", pyint)
        .def(py::init<std::string, float, float, float, float, bool>(),
             "name"_a, "rc"_a, "epsilon"_a, "sigma"_a, "maxForce"_a, "objectAware"_a)
        .def("setSpecificPair", &InteractionLJ::setSpecificPair, 
            "pv1"_a, "pv2"_a, "epsilon"_a, "sigma"_a, "maxForce"_a);
        
    
    //   x0, p, ka, kb, kd, kv, gammaC, gammaT, kbT, mpow, theta, totArea0, totVolume0;
    py::class_<MembraneParameters>(m, "MembraneParameters")
        .def(py::init<>())
        .def_readwrite("x0",        &MembraneParameters::x0)
        .def_readwrite("p",         &MembraneParameters::p)
        .def_readwrite("ka",        &MembraneParameters::ka)
        .def_readwrite("kb",        &MembraneParameters::kb)
        .def_readwrite("kd",        &MembraneParameters::kd)
        .def_readwrite("kv",        &MembraneParameters::kv)
        .def_readwrite("gammaC",    &MembraneParameters::gammaC)
        .def_readwrite("gammaT",    &MembraneParameters::gammaT)
        .def_readwrite("kbT",       &MembraneParameters::kbT)
        .def_readwrite("mpow",      &MembraneParameters::mpow)
        .def_readwrite("theta",     &MembraneParameters::theta)
        .def_readwrite("totArea",   &MembraneParameters::totArea0)
        .def_readwrite("totVolume", &MembraneParameters::totVolume0);
        
    py::class_<InteractionMembrane>(m, "MembraneForces", pyint)
        .def(py::init<std::string, MembraneParameters, bool, float>(),
             "name"_a, "params"_a, "stressFree"_a, "growUntilTime"_a=0);
}

