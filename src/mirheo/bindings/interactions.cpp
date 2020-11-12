// Copyright 2020 ETH Zurich. All Rights Reserved.
#include <mirheo/core/pvs/particle_vector.h>

#include <mirheo/core/interactions/pairwise/base_pairwise.h>
#include <mirheo/core/interactions/factory.h>
#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/interactions/membrane/base_membrane.h>
#include <mirheo/core/interactions/obj_binding.h>
#include <mirheo/core/interactions/obj_rod_binding.h>
#include <mirheo/core/interactions/rod/base_rod.h>

#include "bindings.h"
#include "class_wrapper.h"
#include "variant_cast.h"

namespace mirheo
{

using namespace pybind11::literals;

static std::map<std::string, interaction_factory::VarParam>
castToMap(const py::kwargs& kwargs, const std::string& intName)
{
    std::map<std::string, interaction_factory::VarParam> parameters;

    for (const auto& item : kwargs) {
        std::string key;
        try {
            key = py::cast<std::string>(item.first);
        }
        catch (const py::cast_error& e) {
            die("Could not cast one of the arguments in interaction '%s' to string", intName.c_str());
        }
        try {
            parameters[key] = py::cast<interaction_factory::VarParam>(item.second);
        }
        catch (const py::cast_error& e) {
            die("Could not cast argument '%s' in interaction '%s': wrong type", key.c_str(), intName.c_str());
        }
    }
    return parameters;
}

static std::shared_ptr<BaseMembraneInteraction>
createInteractionMembrane(const MirState *state, std::string name,
                          std::string shearDesc, std::string bendingDesc, std::string filterDesc,
                          bool stressFree, py::kwargs kwargs)
{
    auto parameters = castToMap(kwargs, name);

    return interaction_factory::createInteractionMembrane
        (state, name, shearDesc, bendingDesc, filterDesc, parameters, stressFree);
}

static std::shared_ptr<BaseRodInteraction>
createInteractionRod(const MirState *state, std::string name, std::string stateUpdateDesc, bool dumpEnergies, py::kwargs kwargs)
{
    auto parameters = castToMap(kwargs, name);

    return interaction_factory::createInteractionRod(state, name, stateUpdateDesc, dumpEnergies, parameters);
}

static std::shared_ptr<BasePairwiseInteraction>
createPairwiseInteraction(const MirState *state, const std::string& name,
                          real rc, const std::string& kind, py::kwargs kwargs)
{
    auto parameters = castToMap(kwargs, name);
    return interaction_factory::createPairwiseInteraction(state, name, rc, kind, parameters);
}

void exportInteractions(py::module& m)
{
    py::handlers_class<Interaction> pyInt(m, "Interaction", "Base interaction class");

    py::handlers_class<BasePairwiseInteraction> pyIntPairwise(m, "Pairwise", pyInt, R"(
        Generic pairwise interaction class.
        Can be applied between any kind of :any:`ParticleVector` classes.
        The following interactions are currently implemented:


        * **DPD**:
            Pairwise interaction with conservative part and dissipative + random part acting as a thermostat, see [Groot1997]_

            .. math::

                \mathbf{F}_{ij} &= \left(\mathbf{F}^C_{ij} + \mathbf{F}^D_{ij} + \mathbf{F}^R_{ij} \right)  \mathbf{\hat r} \\
                F^C_{ij} &= \begin{cases} a(1-\frac{r}{r_c}), & r < r_c \\ 0, & r \geqslant r_c \end{cases} \\
                F^D_{ij} &= -\gamma w^2(\tfrac{r}{r_c}) (\mathbf{\hat r} \cdot \mathbf{u}) \\
                F^R_{ij} &= \sigma w(\tfrac{r}{r_c}) \, \frac{\theta}{\sqrt{\Delta t}} \,

            where bold symbol means a vector, its regular counterpart means vector length:
            :math:`x = \left\lVert \mathbf{x} \right\rVert`, hat-ed symbol is the normalized vector:
            :math:`\mathbf{\hat x} = \mathbf{x} / \left\lVert \mathbf{x} \right\rVert`. Moreover, :math:`\theta` is the random variable with zero mean
            and unit variance, that is distributed independently of the interacting pair *i*-*j*, dissipation and random forces
            are related by the fluctuation-dissipation theorem: :math:`\sigma^2 = 2 \gamma \, k_B T`; and :math:`w(r)` is the weight function
            that we define as follows:

            .. math::

                w(r) = \begin{cases} (1-r)^{p}, & r < 1 \\ 0, & r \geqslant 1 \end{cases}


        * **MDPD**:
            Compute MDPD interaction as described in [Warren2003].
            Must be used together with "Density" interaction with kernel "MDPD".

            The interaction forces are the same as described in "DPD" with the modified conservative term

            .. math::

                F^C_{ij} = a w_c(r_{ij}) + b (\rho_i + \rho_j) w_d(r_{ij}),

            where :math:`\rho_i` is computed from "Density" and

            .. math::

                w_c(r) = \begin{cases} (1-\frac{r}{r_c}), & r < r_c \\ 0, & r \geqslant r_c \end{cases} \\
                w_d(r) = \begin{cases} (1-\frac{r}{r_d}), & r < r_d \\ 0, & r \geqslant r_d \end{cases}


        * **SDPD**:
            Compute SDPD interaction with angular momentum conservation, following [Hu2006]_ and [Bian2012]_.
            Must be used together with "Density" interaction with the same density kernel.

            .. math::

                \mathbf{F}_{ij} &= \left(F^C_{ij} + F^D_{ij} + F^R_{ij} \right) \\
                F^C_{ij} &= - \left( \frac{p_{i}}{d_{i}^{2}}+\frac{p_{j}}{d_{j}^{2}}\right) \frac{\partial w_\rho}{\partial r_{ij}}, \\
                F^D_{ij} &= - \eta \left[ \left(\frac{1}{d_{i}^{2}}+\frac{1}{d_{j}^{2}}\right) \frac{-\zeta}{r_{ij}} \frac{\partial w_\rho}{\partial r_{ij}}\right] \left( \mathbf{v}_{i j} \cdot \mathbf{e}_{ij} \right), \\
                F^R_{ij} &= \sqrt{2 k_BT \eta} \left[ \left(\frac{1}{d_{i}^{2}}+\frac{1}{d_{j}^{2}}\right) \frac{-\zeta}{r_{ij}} \frac{\partial w_\rho}{\partial r_{ij}}\right]^{\frac 1 2} \xi_{ij},

            where :math:`\eta` is the viscosity, :math:`w_\rho` is the density kernel, :math:`\zeta = 2+d = 5`, :math:`d_i` is the density of particle i and :math:`p_i = p(d_i)` is the pressure of particle i..
            The available density kernels are listed in "Density".
            The available equations of state (EOS) are:

            Linear equation of state:

                .. math::

                    p(\rho) = c_S^2 \left(\rho - \rho_0 \right)

                where :math:`c_S` is the speed of sound and :math:`\rho_0` is a parameter.

            Quasi incompressible EOS:

                .. math::

                    p(\rho) = p_0 \left[ \left( \frac {\rho}{\rho_r} \right)^\gamma - 1 \right],

                where :math:`p_0`, :math:`\rho_r` and :math:`\gamma = 7` are parameters to be fitted to the desired fluid.


        * **LJ**:
            Pairwise interaction according to the classical `Lennard-Jones potential <https://en.wikipedia.org/wiki/Lennard-Jones_potential>`_

            .. math::

                \mathbf{F}_{ij} = 24 \epsilon \left( 2\left( \frac{\sigma}{r_{ij}} \right)^{12} - \left( \frac{\sigma}{r_{ij}} \right)^{6} \right) \frac{\mathbf{r}}{r^2}

            As opposed to ``RepulsiveLJ``, the force is not bounded from either sides.

        * **RepulsiveLJ**:
            Pairwise interaction according to the classical `Lennard-Jones potential <https://en.wikipedia.org/wiki/Lennard-Jones_potential>`_, truncated such that it is *always repulsive*.

            .. math::

                \mathbf{F}_{ij} = \max \left[ 0.0, 24 \epsilon \left( 2\left( \frac{\sigma}{r_{ij}} \right)^{12} - \left( \frac{\sigma}{r_{ij}} \right)^{6} \right) \frac{\mathbf{r}}{r^2} \right]

            Note that in the implementation, the force is bounded for stability at larger time steps.

        * **Density**:
            Compute density of particles with a given kernel.

            .. math::

                \rho_i = \sum\limits_{j \neq i} w_\rho (r_{ij})

            where the summation goes over the neighbours of particle :math:`i` within a cutoff range of :math:`r_c`.
            The implemented densities are listed below:


            * kernel "MDPD":

                see [Warren2003]_

                .. math::

                    w_\rho(r) = \begin{cases} \frac{15}{2\pi r_d^3}\left(1-\frac{r}{r_d}\right)^2, & r < r_d \\ 0, & r \geqslant r_d \end{cases}

            * kernel "WendlandC2":

                .. math::

                    w_\rho(r) = \frac{21}{2\pi} \left( 1 - \frac{r}{r_c} \right)^4 \left( 1 + 4 \frac{r}{r_c} \right)


        .. [Groot1997] Groot, R. D., & Warren, P. B. (1997).
            Dissipative particle dynamics: Bridging the gap between atomistic and mesoscopic simulations.
            J. Chem. Phys., 107(11), 4423-4435. `doi <https://doi.org/10.1063/1.474784>`

        .. [Warren2003] Warren, P. B.
            "Vapor-liquid coexistence in many-body dissipative particle dynamics."
            Physical Review E 68.6 (2003): 066702.

        .. [Hu2006] Hu, X. Y., and N. A. Adams.
            "Angular-momentum conservative smoothed particle dynamics for incompressible viscous flows."
            Physics of Fluids 18.10 (2006): 101702.

        .. [Bian2012] Bian, Xin, et al.
            "Multiscale modeling of particle in suspension with smoothed dissipative particle dynamics."
            Physics of Fluids 24.1 (2012): 012002.
    )");

    pyIntPairwise.def(py::init(&createPairwiseInteraction),
                 "state"_a, "name"_a, "rc"_a, "kind"_a, R"(
            Args:
                name: name of the interaction
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                kind: interaction kind (e.g. DPD). See below for all possibilities.

            Create one pairwise interaction handler of kind **kind**.
            When applicable, stress computation is activated by passing **stress = True**.
            This activates virial stress computation every **stress_period** time units (also passed in **kwars**)

            * **kind** = "DPD"

                * **a**: :math:`a`
                * **gamma**: :math:`\gamma`
                * **kBT**: :math:`k_B T`
                * **power**: :math:`p` in the weight function

            * **kind** = "MDPD"

                * **rd**: :math:`r_d`
                * **a**: :math:`a`
                * **b**: :math:`b`
                * **gamma**: :math:`\gamma`
                * **kBT**: temperature :math:`k_B T`
                * **power**: :math:`p` in the weight function


            * **kind** = "SDPD"

                * **viscosity**: fluid viscosity
                * **kBT**: temperature :math:`k_B T`
                * **EOS**: the desired equation of state (see below)
                * **density_kernel**: the desired density kernel (see below)


            * **kind** = "RepulsiveLJ"

                * **epsilon**: :math:`\varepsilon`
                * **sigma**: :math:`\sigma`
                * **max_force**: force magnitude will be capped to not exceed **max_force**
                * **aware_mode**:
                    * if "None", all particles interact with each other.
                    * if "Object", the particles belonging to the same object in an object vector do not interact with each other.
                      That restriction only applies if both Particle Vectors in the interactions are the same and is actually an Object Vector.
                    * if "Rod", the particles interact with all other particles except with the ones which are below a given a distance
                      (in number of segment) of the same rod vector. The distance is specified by the kwargs parameter **min_segments_distance**.


            * **kind** = "Density"

                * **density_kernel**: the desired density kernel (see below)

            The available density kernels are "MDPD" and "WendlandC2". Note that "MDPD" can not be used with SDPD interactions.
            MDPD interactions can use only "MDPD" density kernel.

            For SDPD, the available equation of states are given below:

            * **EOS** = "Linear" parameters:

                * **sound_speed**: the speed of sound
                * **rho_0**: background pressure in :math:`c_S` units

            * **EOS** = "QuasiIncompressible" parameters:

                * **p0**: :math:`p_0`
                * **rho_r**: :math:`\rho_r`
    )");

    pyIntPairwise.def("setSpecificPair", [](BasePairwiseInteraction *self, ParticleVector *pv1, ParticleVector *pv2, py::kwargs kwargs)
    {
        auto params = castToMap(kwargs, self->getName());
        self->setSpecificPair(pv1->getName(), pv2->getName(), params);
    }, "pv1"_a, "pv2"_a, R"(
        Set specific parameters of a given interaction for a specific pair of :any:`ParticleVector`.
        This is useful when interactions only slightly differ between different pairs of :any:`ParticleVector`.
        The specific parameters should be set in the **kwargs** field, with same naming as in construction of the interaction.
        Note that only the values of the parameters can be modified, not the kernel types (e.g. change of density kernel is not supported in the case of SDPD interactions).

        Args:
            pv1: first :any:`ParticleVector`
            pv2: second :any:`ParticleVector`
    )");

    py::handlers_class<BaseMembraneInteraction> pyMembraneForces(m, "MembraneForces", pyInt, R"(
        Abstract class for membrane interactions.
        Mesh-based forces acting on a membrane according to the model in [Fedosov2010]_

        The membrane interactions are composed of forces comming from:
            - bending of the membrane, potential :math:`U_b`
            - shear elasticity of the membrane, potential :math:`U_s`
            - constraint: area conservation of the membrane (local and global), potential :math:`U_A`
            - constraint: volume of the cell (assuming incompressible fluid), potential :math:`U_V`
            - membrane viscosity, pairwise force :math:`\mathbf{F}^v`
            - membrane fluctuations, pairwise force :math:`\mathbf{F}^R`

        The form of the constraint potentials are given by (see [Fedosov2010]_ for more explanations):

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

        It is improved with the area-difference model (see [Bian2020]_), which is a discretized version of:

        .. math::

            U_{AD} = \frac{k_{AD} \pi}{2 D_0^2 A_0} \left(\Delta A - \Delta A_0 \right)^2.

        Currently, the stretching and shear energy models are:

        WLC model:

        .. math::

            U_s = \sum_{j \in {1 ... N_s}} \left[ \frac {k_s l_m \left( 3x_j^2 - 2x_j^3 \right)}{4(1-x_j)} + \frac{k_p}{l_0} \right].

        Lim model: an extension of the Skalak shear energy (see [Lim2008]_).

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

        .. [Bian2020] Bian, Xin, Sergey Litvinov, and Petros Koumoutsakos.
                      Bending models of lipid bilayer membranes: Spontaneous curvature and area-difference elasticity.
                      Computer Methods in Applied Mechanics and Engineering 359 (2020): 112758.

        .. [Lim2008] Lim HW, Gerald, Michael Wortis, and Ranjan Mukhopadhyay.
                     Red blood cell shapes and shape transformations: newtonian mechanics of a composite membrane: sections 2.1–2.4.
                     Soft Matter: Lipid Bilayers and Red Blood Cells 4 (2008): 83-139.
    )");

    pyMembraneForces.def(py::init(&createInteractionMembrane),
                         "state"_a, "name"_a,
                         "shear_desc"_a, "bending_desc"_a, "filter_desc"_a = "keep_all",
                         "stress_free"_a=false, R"(
             Args:
                 name: name of the interaction
                 shear_desc: a string describing what shear force is used
                 bending_desc: a string describing what bending force is used
                 filter_desc: a string describing which membranes are concerned
                 stress_free: if True, stress Free shape is used for the shear parameters

             kwargs:

                 * **tot_area**:                total area of the membrane at equilibrium
                 * **tot_volume**:              total volume of the membrane at equilibrium
                 * **ka_tot**:                  constraint energy for total area
                 * **kv_tot**:                  constraint energy for total volume
                 * **kBT**:                     fluctuation temperature (set to zero will switch off fluctuation forces)
                 * **gammaC**:                  central component of dissipative forces
                 * **gammaT**:                  tangential component of dissipative forces (warning: if non zero, the interaction will NOT conserve angular momentum)
                 * **initial_length_fraction**: the size of the membrane increases linearly in time from this fraction of the provided mesh to its full size after grow_until time; the parameters are scaled accordingly with time. If this is set, **grow_until** must also be provided. Default value: 1.
                 * **grow_until**:              the size increases linearly in time from a fraction of the provided mesh to its full size after that time; the parameters are scaled accordingly with time. If this is set, **initial_length_fraction** must also be provided. Default value: 0

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
                 * **DA0**: area difference at relaxed state divided by the offset of the leaflet midplanes

             **filter_desc** = "keep_all":

                 The interaction will be applied to all membranes

             **filter_desc** = "by_type_id":

                 The interaction will be applied membranes with a given **type_id** (see :class:`~libmirheo.InitialConditions.MembraneWithTypeId`)

                 * **type_id**: the type id that the interaction applies to
    )");


    py::handlers_class<ObjectBindingInteraction> pyObjBinding(m, "ObjBinding", pyInt, R"(
        Forces attaching a :any:`ParticleVector` to another via harmonic potentials between the particles of specific pairs.

        .. warning::
            To deal with MPI, the force is zero if two particles of a pair are apart from more than half the subdomain size. Since this interaction is designed to bind objects to each other, this should not happen under normal conditions.

    )");

    pyObjBinding.def(py::init(&interaction_factory::createInteractionObjBinding),
                     "state"_a, "name"_a, "k_bound"_a, "pairs"_a, R"(
            Args:
                name: Name of the interaction.
                k_bound: Spring force coefficient.
                pairs: The global Ids of the particles that will interact through the harmonic potential. For each pair, the first entry is the id of pv1 while the second is that of pv2 (see :any:`libmirheo.setInteraction`).

    )");

    py::handlers_class<ObjectRodBindingInteraction> pyObjRodBinding(m, "ObjRodBinding", pyInt, R"(
        Forces attaching a :any:`RodVector` to a :any:`RigidObjectVector`.
    )");

    pyObjRodBinding.def(py::init(&interaction_factory::createInteractionObjRodBinding),
                        "state"_a, "name"_a, "torque"_a, "rel_anchor"_a, "k_bound"_a, R"(
            Args:
                name: name of the interaction
                torque: torque magnitude to apply to the rod
                rel_anchor: position of the anchor relative to the rigid object
                k_bound: anchor harmonic potential magnitude

    )");


    py::handlers_class<BaseRodInteraction> pyRodForces(m, "RodForces", pyInt, R"(
        Forces acting on an elastic rod.

        The rod interactions are composed of forces comming from:
            - bending energy, :math:`E_{\text{bend}}`
            - twist energy, :math:`E_{\text{twist}}`
            - bounds energy,  :math:`E_{\text{bound}}`

        The form of the bending energy is given by (for a bi-segment):

        .. math::

            E_{\mathrm{bend}}=\frac{l}{4} \sum_{j=0}^{1}\left(\kappa^{j}-\overline{\kappa}\right)^{T} B\left(\kappa^{j}-\overline{\kappa}\right),

        where

        .. math::

            \kappa^{j}=\frac {1} {l} \left((\kappa \mathbf{b}) \cdot \mathbf{m}_{2}^{j},-(\kappa \mathbf{b}) \cdot \mathbf{m}_{1}^{j}\right).

        See, e.g. [bergou2008]_ for more details.
        The form of the twist energy is given by (for a bi-segment):

        .. math::

            E_{\mathrm{twist}}=\frac{k_{t} l}{2}\left(\frac{\theta^{1}-\theta^{0}}{l}-\overline{\tau}\right)^{2}.

        The additional bound energy is a simple harmonic potential with a given equilibrium length.

        .. [bergou2008] Bergou, M.; Wardetzky, M.; Robinson, S.; Audoly, B. & Grinspun, E.
                        Discrete elastic rods
                        ACM transactions on graphics (TOG), 2008, 27, 63

    )");

    pyRodForces.def(py::init(&createInteractionRod),
                    "state"_a, "name"_a, "state_update"_a="none", "save_energies"_a=false, R"(
             Args:
                 name: name of the interaction
                 state_update: description of the state update method; only makes sense for multiple states. See below for possible choices.
                 save_energies: if `True`, save the energies of each bisegment

             kwargs:

                 * **a0** (real):           equilibrium length between 2 opposite cross vertices
                 * **l0** (real):           equilibrium length between 2 consecutive vertices on the centerline
                 * **k_s_center** (real):   elastic force magnitude for centerline
                 * **k_s_frame** (real):    elastic force magnitude for material frame particles
                 * **k_bending** (real3):   Bending symmetric tensor :math:`B` in the order :math:`\left(B_{xx}, B_{xy}, B_{zz} \right)`
                 * **kappa0** (real2):      Spontaneous curvatures along the two material frames :math:`\overline{\kappa}`
                 * **k_twist** (real):      Twist energy magnitude :math:`k_\mathrm{twist}`
                 * **tau0** (real):         Spontaneous twist :math:`\overline{\tau}`
                 * **E0** (real):           (optional) energy ground state

             state update parameters, for **state_update** = 'smoothing':

                 (not fully implemented yet; for now just takes minimum state but no smoothing term)

             state update parameters, for **state_update** = 'spin':

                 * **nsteps** number of MC step per iteration
                 * **kBT** temperature used in the acceptance-rejection algorithm
                 * **J** neighbouring spin 'dislike' energy

             The interaction can support multiple polymorphic states if **kappa0**, **tau0** and **E0** are lists of equal size.
             In this case, the **E0** parameter is required.
             Only lists of 1, 2 and 11 states are supported.
    )");
}


} // namespace mirheo
