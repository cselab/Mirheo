#include <pybind11/pybind11.h>

#include <core/pvs/particle_vector.h>

#include <core/interactions/interface.h>
#include <core/interactions/dpd.h>
#include <core/interactions/lj.h>
#include <core/interactions/membrane.h>

#include "nodelete.h"

namespace py = pybind11;
using namespace pybind11::literals;

void exportInteractions(py::module& m)
{
    // Initial Conditions
    py::handlers_class<Interaction> pyint(m, "Interaction", "Base interaction class");

    py::handlers_class<InteractionDPD>(m, "DPD", pyint, R"(
        Pairwise interaction with conservative part and dissipative + random part acting as a thermostat, see https://aip.scitation.org/doi/abs/10.1063/1.474784
    
        .. math::
        
            \mathbf{F}_{ij} &= \mathbf{F}^C(\mathbf{r}_{ij}) + \mathbf{F}^D(\mathbf{r}_{ij}, \mathbf{u}_{ij}) + \mathbf{F}^R(\mathbf{r}_{ij}) \\
            \mathbf{F}^C(\mathbf{r}) &= \begin{cases} a(1-\frac{r}{r_c}) \mathbf{\hat r}, & r < r_c \\ 0, & r \geqslant r_c \end{cases} \\
            \mathbf{F}^D(\mathbf{r}, \mathbf{u}) &= \gamma w^2(\frac{r}{r_c}) (\mathbf{r} \cdot \mathbf{u}) \mathbf{\hat r} \\
            \mathbf{F}^R(\mathbf{r}) &= \sigma w(\frac{r}{r_c}) \, \theta \sqrt{\Delta t} \, \mathbf{\hat r}
        
        where bold symbol means a vector, its regular counterpart means vector length: 
        :math:`x = \left\lVert \mathbf{x} \right\rVert`, hat-ed symbol is the normalized vector:
        :math:`\mathbf{\hat x} = \mathbf{x} / \left\lVert \mathbf{x} \right\rVert`. Moreover, :math:`\theta` is the random variable with zero mean
        and unit variance, that is distributed independently of the interacting pair *i*-*j*, dissipation and random forces 
        are related by the fluctuation-dissipation theorem: :math:`\sigma^2 = 2 \gamma \, k_B T`; and :math:`w(r)` is the weight function
        that we define as follows:
        
        .. math::
            
            w(r) = \begin{cases} (1-r)^{p}, & r < 1 \\ 0, & r \geqslant 1 \end{cases}
    )")
        .def(py::init<std::string, float, float, float, float, float, float>(),
             "name"_a, "rc"_a, "a"_a, "gamma"_a, "kbt"_a, "dt"_a, "power"_a, R"(  
                Args:
                    name: name of the interaction
                    rc: interaction cut-off (no forces between particles further than **rc** apart)
                    a: :math:`a`
                    gamma: :math:`\gamma`
                    kbt: :math:`k_B T`
                    dt: time-step, that for consistency has to be the same as the integration time-step for the corresponding particle vectors
                    power: :math:`p` in the weight function
        )")
        .def("setSpecificPair", &InteractionDPD::setSpecificPair, 
            "pv1"_a, "pv2"_a, "a"_a, "gamma"_a, "kbt"_a, "dt"_a, "power"_a, R"(
                Override some of the interaction parameters for a specific pair of Particle Vectors
            )");
        
    py::handlers_class<InteractionLJ>(m, "LJ", pyint, R"(
        Pairwise interaction according to the classical Lennard-Jones potential `http://rspa.royalsocietypublishing.org/content/106/738/463`
        The force however is truncated such that it is *always repulsive*.
        
        .. math::
        
            \mathbf{F}_{ij} = \max \left[ 0.0, 24 \epsilon \left( 2\left( \frac{\sigma}{r_{ij}} \right)^{14} - \left( \frac{\sigma}{r_{ij}} \right)^{8} \right) \right]
   
    )")
        .def(py::init<std::string, float, float, float, float, bool>(),
             "name"_a, "rc"_a, "epsilon"_a, "sigma"_a, "max_force"_a, "object_aware"_a, R"(
                Args:
                    name: name of the interaction
                    rc: interaction cut-off (no forces between particles further than **rc** apart)
                    epsilon: :math:`\varepsilon`
                    sigma: :math:`\sigma`
                    max_force: force magnitude will be capped not exceed **max_force**
                    object_aware:
                        if True, the particles belonging to the same object in an object vector do not interact with each other.
                        That restriction only applies if both Particle Vectors in the interactions are the same and is actually an Object Vector. 
        )")
        .def("setSpecificPair", &InteractionLJ::setSpecificPair, 
            "pv1"_a, "pv2"_a, "epsilon"_a, "sigma"_a, "maxForce"_a, R"(
                Override some of the interaction parameters for a specific pair of Particle Vectors
            )");
        
    
    //   x0, p, ka, kb, kd, kv, gammaC, gammaT, kbT, mpow, theta, totArea0, totVolume0;
    py::handlers_class<MembraneParameters>(m, "MembraneParameters")
        .def(py::init<>(), R"(
            Structure keeping parameters of the membrane interaction
        )")
        .def_readwrite("x0",        &MembraneParameters::x0)
        .def_readwrite("p",         &MembraneParameters::p)
        .def_readwrite("ks",        &MembraneParameters::ks)
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
        
    py::handlers_class<InteractionMembrane>(m, "MembraneForces", pyint, R"(
        Mesh-based forces acting on a membrane according to the model in PUT LINK
    )")
        .def(py::init<std::string, MembraneParameters, bool, float>(),
             "name"_a, "params"_a, "stressFree"_a, "growUntilTime"_a=0, R"( TODO
        )");
}

