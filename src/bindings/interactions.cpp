#include <core/pvs/particle_vector.h>

#include <core/interactions/interface.h>
#include <core/interactions/dpd.h>
#include <core/interactions/dpd_with_stress.h>
#include <core/interactions/mdpd.h>
#include <core/interactions/mdpd_with_stress.h>
#include <core/interactions/lj.h>
#include <core/interactions/lj_with_stress.h>
#include <core/interactions/membrane_WLC_Kantor.h>
#include <core/interactions/membrane_WLC_Juelicher.h>
#include <core/interactions/factory.h>

#include "bindings.h"
#include "class_wrapper.h"

using namespace pybind11::literals;

static std::shared_ptr<InteractionMembrane>
createInteractionMembrane(const YmrState *state, std::string name,
                          std::string shearDesc, std::string bendingDesc,
                          bool stressFree, float growUntil, py::kwargs kwargs)
{
    std::map<std::string, float> parameters;

    for (const auto& item : kwargs) {
        try {
            auto key   = py::cast<std::string>(item.first);
            auto value = py::cast<float>(item.second);
            parameters[key] = value;
        }
        catch (const py::cast_error& e)
        {
            die("Could not cast one of the arguments in membrane interactions");
        }        
    }    
    
    return InteractionFactory::createInteractionMembrane
        (state, name, shearDesc, bendingDesc, parameters, stressFree, growUntil);
}


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
            J. Chem. Phys., 107(11), 4423-4435. `doi <https://doi.org/10.1063/1.474784>`_
    )");

    pyIntDPD.def(py::init<const YmrState*, std::string, float, float, float, float, float>(),
                 "state"_a, "name"_a, "rc"_a, "a"_a, "gamma"_a, "kbt"_a, "power"_a, R"(  
            Args:
                name: name of the interaction
                    rc: interaction cut-off (no forces between particles further than **rc** apart)
                    a: :math:`a`
                    gamma: :math:`\gamma`
                    kbt: :math:`k_B T`
                    power: :math:`p` in the weight function
    )");

    pyIntDPD.def("setSpecificPair", &InteractionDPD::setSpecificPair, 
         "pv1"_a, "pv2"_a,
         "a"_a=InteractionDPD::Default, "gamma"_a=InteractionDPD::Default,
         "kbt"_a=InteractionDPD::Default, "power"_a=InteractionDPD::Default,
         R"(
            Override some of the interaction parameters for a specific pair of Particle Vectors
         )");
        
    py::handlers_class<InteractionDPDWithStress> pyIntDPDWithStress(m, "DPDWithStress", pyIntDPD, R"(
        wrapper of :any:`DPD` with, in addition, stress computation
    )");

    pyIntDPDWithStress.def(py::init<const YmrState*, std::string, float, float, float, float, float, float>(),
                           "state"_a, "name"_a, "rc"_a, "a"_a, "gamma"_a, "kbt"_a, "power"_a, "stressPeriod"_a, R"(  
            Args:
                name: name of the interaction
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                a: :math:`a`
                gamma: :math:`\gamma`
                kbt: :math:`k_B T`
                power: :math:`p` in the weight function
                stressPeriod: compute the stresses every this period (in simulation time units)
    )");

    py::handlers_class<InteractionDensity> pyIntDensity(m, "Density", pyInt, R"(
        Compute MDPD density of particles, see [Warren2003]_
    
        .. math::
        
            \rho_i = \sum\limits_{j \neq i} w_\rho (r_{ij})

        where the summation goes over the neighbours of particle :math:`i` within a cutoff range of :math:`r_c`, and

        .. math::
            
            w_\rho(r) = \begin{cases} \frac{15}{2\pi r_d^3}\left(1-\frac{r}{r_d}\right)^2, & r < r_d \\ 0, & r \geqslant r_d \end{cases}            
    )");
    
    pyIntDensity.def(py::init<const YmrState*, std::string, float>(),
                     "state"_a, "name"_a, "rc"_a, R"(  
            Args:
                name: name of the interaction
                rc: interaction cut-off
    )");

    py::handlers_class<InteractionMDPD> pyIntMDPD(m, "MDPD", pyInt, R"(
        Compute MDPD interaction as described in [Warren2003].
        Must be used together with :any:`Density` interaction.

        The interaction forces are the same as described in :any:`DPD` with the modified conservative term

        .. math::

            F^C_{ij} = a w_c(r_{ij}) + b (\rho_i + \rho_j) w_d(r_{ij}),
 
        where :math:`\rho_i` is computed from :any:`Density` and

        .. math::

            w_c(r) = \begin{cases} (1-\frac{r}{r_c}), & r < r_c \\ 0, & r \geqslant r_c \end{cases} \\
            w_d(r) = \begin{cases} (1-\frac{r}{r_d}), & r < r_d \\ 0, & r \geqslant r_d \end{cases}


        .. [Warren2003] Warren, P. B. 
           "Vapor-liquid coexistence in many-body dissipative particle dynamics."
           Physical Review E 68.6 (2003): 066702.`_
    )");
    
    pyIntMDPD.def(py::init<const YmrState*, std::string, float, float, float, float, float, float, float>(),
                  "state"_a, "name"_a, "rc"_a, "rd"_a, "a"_a, "b"_a, "gamma"_a, "kbt"_a, "power"_a, R"(  
            Args:
                name: name of the interaction
                    rc: interaction cut-off (no forces between particles further than **rc** apart)
                    rd: density cutoff, assumed rd <= rc
                    a: :math:`a`
                    b: :math:`b`
                    gamma: :math:`\gamma`
                    kbt: :math:`k_B T`
                    power: :math:`p` in the weight function
    )");

    py::handlers_class<InteractionMDPDWithStress> pyIntMDPDWithStress(m, "MDPDWithStress", pyIntMDPD, R"(
        wrapper of :any:`MDPD` with, in addition, stress computation
    )");

    pyIntMDPDWithStress.def(py::init<const YmrState*, std::string, float, float, float, float, float, float, float, float>(),
                            "state"_a, "name"_a, "rc"_a, "rd"_a, "a"_a, "b"_a, "gamma"_a, "kbt"_a, "power"_a, "stressPeriod"_a, R"(  
            Args:
                name: name of the interaction
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                rd: density cut-off, assumed rd < rc
                a: :math:`a`
                b: :math:`b`
                gamma: :math:`\gamma`
                kbt: :math:`k_B T`
                power: :math:`p` in the weight function
                stressPeriod: compute the stresses every this period (in simulation time units)
    )");


    py::handlers_class<InteractionLJ> pyIntLJ (m, "LJ", pyInt, R"(
        Pairwise interaction according to the classical `Lennard-Jones potential <https://en.wikipedia.org/wiki/Lennard-Jones_potential>`_
        The force however is truncated such that it is *always repulsive*.
        
        .. math::
        
            \mathbf{F}_{ij} = \max \left[ 0.0, 24 \epsilon \left( 2\left( \frac{\sigma}{r_{ij}} \right)^{14} - \left( \frac{\sigma}{r_{ij}} \right)^{8} \right) \right]
   
    )");

    
    pyIntLJ.def(py::init<const YmrState*, std::string, float, float, float, float, bool>(),
                "state"_a, "name"_a, "rc"_a, "epsilon"_a, "sigma"_a, "max_force"_a=1000.0, "object_aware"_a, R"(
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

    pyIntLJWithStress.def(py::init<const YmrState*, std::string, float, float, float, float, bool, float>(),
                          "state"_a, "name"_a, "rc"_a, "epsilon"_a, "sigma"_a, "max_force"_a=1000.0,
                          "object_aware"_a, "stressPeriod"_a, R"(
            Args:
                name: name of the interaction
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                epsilon: :math:`\varepsilon`
                sigma: :math:`\sigma`
                max_force: force magnitude will be capped not exceed **max_force**
                object_aware:
                    if True, the particles belonging to the same object in an object vector do not interact with each other.
                    That restriction only applies if both Particle Vectors in the interactions are the same and is actually an Object Vector. 
                stressPeriod: compute the stresses every this period (in simulation time units)
    )");
    
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

        The form of the constrain potentials are given by (see [Fedosov2010]_ for more explanations):

        .. math::

            U_A = \frac{k_a (A_{tot} - A^0_{tot})^2}{2 A^0_{tot}} + \sum_{j \in {1 ... N_t}} \frac{k_d (A_j-A_0)^2}{2A_0}, \\
            U_V = \frac{k_v (V-V^0_{tot})^2}{2 V^0_{tot}}.

        The viscous and dissipation forces are central forces and are the same as DPD interactions with :math:`w(r) = 1` 
        (no cutoff radius, applied to each bond).

        Several bending models are implemented. First, the Kantor enrgy reads (see [kantor1987]_):

        .. math::

            U_b = \sum_{j \in {1 ... N_s}} k_b \left[  1-\cos(\theta_j - \theta_0) \right].

        The Juelicher energy is (see [Juelicher1996]_):

        .. math::

            U_b = 2 k_b \sum_{\alpha = 1}^{N_v} \frac {\left( M_{\alpha} - C_0\right)^2}{A_\alpha}, \\
            M_{\alpha} = \frac 1 4 \sum_{<i,j>}^{(\alpha)} l_{ij} \theta_{ij}.

        It is improve with ADE model (TODO: ref).

        Currently, the stretching and shear energiy models are:
        WLC model:

        .. math::

            U_s = \sum_{j \in {1 ... N_s}} \left[ \frac {k_s l_m \left( 3x_j^2 - 2x_j^3 \right)}{4(1-x_j)} + \frac{k_p}{l_0} \right].

        Lim model, which is an extension of the Skalak shear energy (see [Lim2008]_).

        .. math::
        
            U_{Lim} =& \sum_{i=1}^{N_{t}}\left(A_{0}\right)_{i}\left(\frac{k_a}{2}\left(\alpha_{i}^{2}+a_{3} \alpha_{i}^{3}+a_{4} \alpha_{i}^{4}\right)\right.\\
                     & +\mu\left(\beta_{i}+b_{1} \alpha_{i} \beta_{i}+b_{2} \beta_{i}^{2}\right) ),

        where :math:`\alpha` and :math:`\beta` are the invariants of the strains.

        .. [Fedosov2010] Fedosov, D. A.; Caswell, B. & Karniadakis, G. E. 
                         A multiscale red blood cell model with accurate mechanics, rheology, and dynamics 
                         Biophysical journal, Elsevier, 2010, 98, 2215-2225

        .. [kantor1987] Kantor, Y. & Nelson, D. R. 
                        Phase transitions in flexible polymeric surfaces 
                        Physical Review A, APS, 1987, 36, 4020

        .. [Juelicher1996] Juelicher, Frank, and Reinhard Lipowsky. 
                           Shape transformations of vesicles with intramembrane domains.
                           Physical Review E 53.3 (1996): 2670.

        .. [Lim2008] Lim HW, Gerald, Michael Wortis, and Ranjan Mukhopadhyay. 
                     Red blood cell shapes and shape transformations: newtonian mechanics of a composite membrane: sections 2.1–2.4.
                     Soft Matter: Lipid Bilayers and Red Blood Cells 4 (2008): 83-139.
    )");

    pyMembraneForces.def(py::init(&createInteractionMembrane),
                         "state"_a, "name"_a, "shear_desc"_a, "bending_desc"_a,
                         "stress_free"_a=false, "grow_until"_a=0.f, R"( 
             Args:
                 name: name of the interaction
                 shear_desc: a string describing what shear force is used
                 bending_desc: a string describing what bending force is used
                 stress_free: if True, stress Free shape is used for the shear parameters
                 grow_until: the size increases linearly in time from half of the provided mesh 
                             to its full size after that time; the parameters are scaled accordingly with time

             kwargs:

                 * **tot_area**:   total area of the membrane at equilibrium
                 * **tot_volume**: total volume of the membrane at equilibrium
                 * **ka_tot**:     constrain energy for total area
                 * **kv_tot**:     constrain energy for total volume
                 * **kBT**:        fluctuation temperature (set to zero will switch off fluctuation forces)
                 * **gammaC**:     central component of dissipative forces
                 * **gammaT**:     tangential component of dissipative forces (warning: if non zero, the interaction will NOT conserve angular momentum)

             Shear Parameters, warm like chain model (set **shear_desc** = 'wlc'):

                 * **x0**:   :math:`x_0`
                 * **ks**:   energy magnitude for bonds
                 * **mpow**: :math:`m`
                 * **ka**:   energy magnitude for local area

             Shear Parameters, Lim model (set **shear_desc** = 'Lim'):

                 * **ka**: :math:`k_a`, magnitude of stretching force
                 * **mu**: :math:`\mu`, magnitude of shear force
                 * **a3**: :math:`a_3`, non linear part for stretching 
                 * **a4**: :math:`a_4`, non linear part for stretching 
                 * **b1**: :math:`b_1`, non linear part for shear
                 * **b2**: :math:`b_2`, non linear part for shear

             Bending Parameters, Kantor model (set **bending_desc** = 'Kantor'):

                 * **kb**:    local bending energy magnitude
                 * **theta**: spontaneous angle

             Bending Parameters, Juelicher model (set **bending_desc** = 'Juelicher'):

                 * **kb**:  local bending energy magnitude
                 * **C0**:  spontaneous curvature
                 * **kad**: area difference energy magnitude
                 * **DA0**: spontaneous area difference
    )");
}

