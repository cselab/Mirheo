#include <core/pvs/particle_vector.h>

#include <core/interactions/interface.h>
#include <core/interactions/dpd.h>
#include <core/interactions/dpd_with_stress.h>
#include <core/interactions/lj.h>
#include <core/interactions/lj_with_stress.h>
#include <core/interactions/membrane_kantor.h>
#include <core/interactions/membrane_juelicher.h>

#include "bindings.h"
#include "class_wrapper.h"

using namespace pybind11::literals;

void exportInteractions(py::module& m)
{
    py::handlers_class<Interaction> pyInt(m, "Interaction", "Base interaction class");

    py::handlers_class<InteractionDPD> pyIntDPD(m, "DPD", pyInt, R"(
        Pairwise interaction with conservative part and dissipative + random part acting as a thermostat, see [Groot1997]_
    
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
            
        .. [Groot1997] Groot, R. D., & Warren, P. B. (1997).
            Dissipative particle dynamics: Bridging the gap between atomistic and mesoscopic simulations.
            J. Chem. Phys., 107(11), 4423–4435. `doi <https://doi.org/10.1063/1.474784>`_
    )");

    pyIntDPD.def(py::init<std::string, const YmrState*, float, float, float, float, float, float>(),
                 "name"_a, "state"_a, "rc"_a, "a"_a, "gamma"_a, "kbt"_a, "dt"_a, "power"_a, R"(  
            Args:
            name: name of the interaction
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                a: :math:`a`
                gamma: :math:`\gamma`
                kbt: :math:`k_B T`
                dt: time-step, that for consistency has to be the same as the integration time-step for the corresponding particle vectors
                power: :math:`p` in the weight function
    )");

    pyIntDPD.def("setSpecificPair", &InteractionDPD::setSpecificPair, 
         "pv1"_a, "pv2"_a,
         "a"_a=InteractionDPD::Default, "gamma"_a=InteractionDPD::Default,
         "kbt"_a=InteractionDPD::Default, "dt"_a=InteractionDPD::Default, "power"_a=InteractionDPD::Default,
         R"(
            Override some of the interaction parameters for a specific pair of Particle Vectors
         )");
        
    py::handlers_class<InteractionDPDWithStress> pyIntDPDWithStress(m, "DPDWithStress", pyIntDPD, R"(
        wrapper of :any:`DPD` with, in addition, stress computation
    )");

    pyIntDPDWithStress.def(py::init<std::string, const YmrState*, std::string, float, float, float, float, float, float, float>(),
                           "name"_a, "state"_a, "stressName"_a, "rc"_a, "a"_a, "gamma"_a, "kbt"_a, "dt"_a, "power"_a, "stressPeriod"_a, R"(  
            Args:
                name: name of the interaction
                stressName: name of the stress entry
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                a: :math:`a`
                gamma: :math:`\gamma`
                kbt: :math:`k_B T`
                dt: time-step, that for consistency has to be the same as the integration time-step for the corresponding particle vectors
                power: :math:`p` in the weight function
                stressPeriod: compute the stresses every this period (in simulation time units)
    )");

    py::handlers_class<InteractionLJ> pyIntLJ (m, "LJ", pyInt, R"(
        Pairwise interaction according to the classical `Lennard-Jones potential <https://en.wikipedia.org/wiki/Lennard-Jones_potential>`_
        The force however is truncated such that it is *always repulsive*.
        
        .. math::
        
            \mathbf{F}_{ij} = \max \left[ 0.0, 24 \epsilon \left( 2\left( \frac{\sigma}{r_{ij}} \right)^{14} - \left( \frac{\sigma}{r_{ij}} \right)^{8} \right) \right]
   
    )");

    pyIntLJ.def(py::init<std::string, const YmrState*, float, float, float, float, bool>(),
                "name"_a, "state"_a, "rc"_a, "epsilon"_a, "sigma"_a, "max_force"_a=1000.0, "object_aware"_a, R"(
            Args:
                name: name of the interaction
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                epsilon: :math:`\varepsilon`
                sigma: :math:`\sigma`
                max_force: force magnitude will be capped not exceed **max_force**
                object_aware:
                    if True, the particles belonging to the same object in an object vector do not interact with each other.
                    That restriction only applies if both Particle Vectors in the interactions are the same and is actually an Object Vector. 
    )");

    pyIntLJ.def("setSpecificPair", &InteractionLJ::setSpecificPair, 
        "pv1"_a, "pv2"_a, "epsilon"_a, "sigma"_a, "max_force"_a, R"(
            Override some of the interaction parameters for a specific pair of Particle Vectors
        )");
        
    py::handlers_class<InteractionLJWithStress> pyIntLJWithStress (m, "LJWithStress", pyIntLJ, R"(
        wrapper of :any:`LJ` with, in addition, stress computation
    )");

    pyIntLJWithStress.def(py::init<std::string, const YmrState*, std::string, float, float, float, float, bool, float>(),
                          "name"_a, "state"_a, "stressName"_a, "rc"_a, "epsilon"_a, "sigma"_a, "max_force"_a=1000.0,
                          "object_aware"_a, "stressPeriod"_a, R"(
            Args:
                name: name of the interaction
                stressName: name of the stress entry
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                epsilon: :math:`\varepsilon`
                sigma: :math:`\sigma`
                max_force: force magnitude will be capped not exceed **max_force**
                object_aware:
                    if True, the particles belonging to the same object in an object vector do not interact with each other.
                    That restriction only applies if both Particle Vectors in the interactions are the same and is actually an Object Vector. 
                stressPeriod: compute the stresses every this period (in simulation time units)
    )");
    
    py::handlers_class<MembraneParameters>(m, "MembraneParameters", R"(
        Common membrane parameters
    )")
        .def(py::init<>(), R"(
            Structure keeping parameters of the membrane interaction
        )")
        .def_readwrite("x0",        &MembraneParameters::x0)
        .def_readwrite("ks",        &MembraneParameters::ks)
        .def_readwrite("ka",        &MembraneParameters::ka)
        .def_readwrite("kd",        &MembraneParameters::kd)
        .def_readwrite("kv",        &MembraneParameters::kv)
        .def_readwrite("gammaC",    &MembraneParameters::gammaC)
        .def_readwrite("gammaT",    &MembraneParameters::gammaT)
        .def_readwrite("kbT",       &MembraneParameters::kbT)
        .def_readwrite("mpow",      &MembraneParameters::mpow)
        .def_readwrite("totArea",   &MembraneParameters::totArea0)
        .def_readwrite("totVolume", &MembraneParameters::totVolume0)
        .def_readwrite("rnd",       &MembraneParameters::fluctuationForces)
        .def_readwrite("dt",        &MembraneParameters::dt);
        
    py::handlers_class<InteractionMembrane> pyMembraneForces(m, "MembraneForces", pyInt, R"(
        Abstract class for membrane interactions.
        Mesh-based forces acting on a membrane according to the model in [Fedosov2010]_

        The membrane interactions are composed of forces comming from:
            - bending of the membrane, potential :math:`U_b`
            - shear elasticity of the membrane, potential :math:`U_s`
            - constrain: area conservation of the membrane (local and global), potential :math:`U_A`
            - constrain: volume of the cell (assuming incompressible fluid), potential :math:`U_V`
            - membrane viscosity, pairwise force :math:`\mathbf{F}^v`
            - membrane fluctuations, pairwise force :math:`\mathbf{F}^R`

        The form of these potentials is given by:

        .. math::

            U_b = \sum_{j \in {1 ... N_s}} k_b \left[  1-\cos(\theta_j - \theta_0) \right], \\
            U_s = \sum_{j \in {1 ... N_s}} \left[ \frac {k_s l_m \left( 3x_j^2 - 2x_j^3 \right)}{4(1-x_j)} + \frac{k_p}{l_0} \right], \\
            U_A = \frac{k_a (A_{tot} - A^0_{tot})^2}{2 A^0_{tot}} + \sum_{j \in {1 ... N_t}} \frac{k_d (A_j-A_0)^2}{2A_0}, \\
            U_V = \frac{k_v (V-V^0_{tot})^2}{2 V^0_{tot}}.

        See [Fedosov2010]_ for more explanations.
        The viscous and dissipation forces are central forces and are the same as DPD interactions with :math:`w(r) = 1` 
        (no cutoff radius, applied to each bond).

        .. [Fedosov2010] Fedosov, D. A.; Caswell, B. & Karniadakis, G. E. 
                             A multiscale red blood cell model with accurate mechanics, rheology, and dynamics 
                             Biophysical journal, Elsevier, 2010, 98, 2215-2225

    )");

    py::handlers_class<KantorBendingParameters>(m, "KantorBendingParameters", R"(
        Bending parameters for Kantor model
    )")
        .def(py::init<>(), R"(
            Structure keeping parameters of the bending membrane interaction
        )")
        .def_readwrite("kb",    &KantorBendingParameters::kb)
        .def_readwrite("theta", &KantorBendingParameters::theta);

    py::handlers_class<JuelicherBendingParameters>(m, "JuelicherBendingParameters", R"(
        Bending parameters for Juelicher model
    )")
        .def(py::init<>(), R"(
            Structure keeping parameters of the bending membrane interaction
        )")
        .def_readwrite("kb",  &JuelicherBendingParameters::kb)
        .def_readwrite("C0",  &JuelicherBendingParameters::C0)
        .def_readwrite("kad", &JuelicherBendingParameters::kad)
        .def_readwrite("DA0", &JuelicherBendingParameters::DA0);


    py::handlers_class<InteractionMembraneKantor> (m, "MembraneForcesKantor", pyInt, R"(
        Mesh-based forces acting on a membrane according to the model in [Fedosov2010]_

         The bending potential :math:`U_b` is defined as:

        .. math::

            U_b = \sum_{j \in {1 ... N_s}} k_b \left[  1-\cos(\theta_j - \theta_0) \right]

        See [Fedosov2010]_ for more explanations.
        The viscous and dissipation forces are central forces and are the same as DPD interactions with :math:`w(r) = 1` 
        (no cutoff radius, applied to each bond).

        .. [Fedosov2010] Fedosov, D. A.; Caswell, B. & Karniadakis, G. E. 
                             A multiscale red blood cell model with accurate mechanics, rheology, and dynamics 
                             Biophysical journal, Elsevier, 2010, 98, 2215-2225

    )")
        .def(py::init<std::string, const YmrState*, MembraneParameters, KantorBendingParameters, bool, float>(),
             "name"_a, "state"_a, "params"_a, "params_bending"_a, "stressFree"_a, "grow_until"_a=0, R"( 
             Args:
                 name: name of the interaction
                 params: instance of :any: `MembraneParameters`
                 params_bending: instance of :any: `KantorBendingParameters`
                 stressFree: equilibrium bond length and areas are taken from the initial mesh
                 grow_until: time to grow the cell at initialization stage; 
                             the size increases linearly in time from half of the provided mesh to its full size after that time
                             the parameters are scaled accordingly with time
    )");

    py::handlers_class<InteractionMembraneJuelicher> (m, "MembraneForcesJuelicher", pyInt, R"(
        Mesh-based forces acting on a membrane according to the model in [Fedosov2010]_ with Juelicher bending model.

        The bending potential :math:`U_b` is defined as:

        .. math::

            U_b = 2 k_b \sum_{\alpha = 1}^{N_v} \frac {\left( M_{\alpha} - C_0\right)^2}{A_\alpha}, \\
            M_{\alpha} = \frac 1 4 \sum_{<i,j>}^{(\alpha)} l_{ij} \theta_{ij}.

        See [Juelicher1996]_ for more explanations. Note that the current model is an extended version of the original form.
        The viscous and dissipation forces are central forces and are the same as DPD interactions with :math:`w(r) = 1` 
        (no cutoff radius, applied to each bond).

        .. [Juelicher1996] Juelicher, Frank, and Reinhard Lipowsky. 
                           Shape transformations of vesicles with intramembrane domains.
                           Physical Review E 53.3 (1996): 2670.
    )")
        .def(py::init<std::string, const YmrState*, MembraneParameters, JuelicherBendingParameters, bool, float>(),
             "name"_a, "state"_a, "params"_a, "params_bending"_a, "stressFree"_a, "grow_until"_a=0, R"( 
             Args:
                 name: name of the interaction
                 params: instance of :any: `MembraneParameters`
                 params_bending: instance of :any: `JuelicherBendingParameters`
                 stressFree: equilibrium bond length and areas are taken from the initial mesh
                 grow_until: time to grow the cell at initialization stage; 
                             the size increases linearly in time from half of the provided mesh to its full size after that time
                             the parameters are scaled accordingly with time
    )");
}

