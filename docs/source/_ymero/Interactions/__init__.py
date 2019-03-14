class Interaction:
    r"""Base interaction class
    """
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
        """
        pass

class DPD(Interaction):
    r"""
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
    
    """
    def __init__():
        r"""__init__(name: str, rc: float, a: float, gamma: float, kbt: float, power: float) -> None

  
            Args:
                name: name of the interaction
                    rc: interaction cut-off (no forces between particles further than **rc** apart)
                    a: :math:`a`
                    gamma: :math:`\gamma`
                    kbt: :math:`k_B T`
                    power: :math:`p` in the weight function
    

        """
        pass

    def setSpecificPair():
        r"""setSpecificPair(pv1: ParticleVectors.ParticleVector, pv2: ParticleVectors.ParticleVector, a: float = inf, gamma: float = inf, kbt: float = inf, power: float = inf) -> None


            Override some of the interaction parameters for a specific pair of Particle Vectors
         

        """
        pass

class LJ(Interaction):
    r"""
        Pairwise interaction according to the classical `Lennard-Jones potential <https://en.wikipedia.org/wiki/Lennard-Jones_potential>`_
        The force however is truncated such that it is *always repulsive*.
        
        .. math::
        
            \mathbf{F}_{ij} = \max \left[ 0.0, 24 \epsilon \left( 2\left( \frac{\sigma}{r_{ij}} \right)^{14} - \left( \frac{\sigma}{r_{ij}} \right)^{8} \right) \right]
   
    
    """
    def __init__():
        r"""__init__(name: str, rc: float, epsilon: float, sigma: float, max_force: float = 1000.0, object_aware: bool) -> None


            Args:
                name: name of the interaction
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                epsilon: :math:`\varepsilon`
                sigma: :math:`\sigma`
                max_force: force magnitude will be capped not exceed **max_force**
                object_aware:
                    if True, the particles belonging to the same object in an object vector do not interact with each other.
                    That restriction only applies if both Particle Vectors in the interactions are the same and is actually an Object Vector. 
    

        """
        pass

    def setSpecificPair():
        r"""setSpecificPair(pv1: ParticleVectors.ParticleVector, pv2: ParticleVectors.ParticleVector, epsilon: float, sigma: float, max_force: float) -> None


            Override some of the interaction parameters for a specific pair of Particle Vectors
        

        """
        pass

class MDPD(Interaction):
    r"""
        Compute MDPD interaction as described in [Warren2003].
        Must be used together with :any:`MDPDDensity` interaction.

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
    
    """
    def __init__():
        r"""__init__(name: str, rc: float, rd: float, a: float, b: float, gamma: float, kbt: float, power: float) -> None

  
            Args:
                name: name of the interaction
                    rc: interaction cut-off (no forces between particles further than **rc** apart)
                    rd: density cutoff, assumed rd <= rc
                    a: :math:`a`
                    b: :math:`b`
                    gamma: :math:`\gamma`
                    kbt: :math:`k_B T`
                    power: :math:`p` in the weight function
    

        """
        pass

class MDPDDensity(Interaction):
    r"""
        Compute MDPD density of particles, see [Warren2003]_
    
        .. math::
        
            \rho_i = \sum\limits_{j \neq i} w_\rho (r_{ij})

        where the summation goes over the neighbours of particle :math:`i` within a cutoff range of :math:`r_c`, and

        .. math::
            
            w_\rho(r) = \begin{cases} \frac{15}{2\pi r_d^3}\left(1-\frac{r}{r_d}\right)^2, & r < r_d \\ 0, & r \geqslant r_d \end{cases}            
    
    """
    def __init__():
        r"""__init__(name: str, rc: float) -> None

  
            Args:
                name: name of the interaction
                rc: interaction cut-off
    

        """
        pass

class MembraneForces(Interaction):
    r"""
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
    
    """
    def __init__():
        r"""__init__(name: str, shear_desc: str, bending_desc: str, stress_free: bool = False, grow_until: float = 0.0, **kwargs) -> None

 
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
    

        """
        pass

class DPDWithStress(DPD):
    r"""
        wrapper of :any:`DPD` with, in addition, stress computation
    
    """
    def __init__():
        r"""__init__(name: str, rc: float, a: float, gamma: float, kbt: float, power: float, stressPeriod: float) -> None

  
            Args:
                name: name of the interaction
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                a: :math:`a`
                gamma: :math:`\gamma`
                kbt: :math:`k_B T`
                power: :math:`p` in the weight function
                stressPeriod: compute the stresses every this period (in simulation time units)
    

        """
        pass

    def setSpecificPair():
        r"""setSpecificPair(pv1: ParticleVectors.ParticleVector, pv2: ParticleVectors.ParticleVector, a: float = inf, gamma: float = inf, kbt: float = inf, power: float = inf) -> None


            Override some of the interaction parameters for a specific pair of Particle Vectors
         

        """
        pass

class LJWithStress(LJ):
    r"""
        wrapper of :any:`LJ` with, in addition, stress computation
    
    """
    def __init__():
        r"""__init__(name: str, rc: float, epsilon: float, sigma: float, max_force: float = 1000.0, object_aware: bool, stressPeriod: float) -> None


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
    

        """
        pass

    def setSpecificPair():
        r"""setSpecificPair(pv1: ParticleVectors.ParticleVector, pv2: ParticleVectors.ParticleVector, epsilon: float, sigma: float, max_force: float) -> None


            Override some of the interaction parameters for a specific pair of Particle Vectors
        

        """
        pass

class MDPDWithStress(MDPD):
    r"""
        wrapper of :any:`MDPD` with, in addition, stress computation
    
    """
    def __init__():
        r"""__init__(name: str, rc: float, rd: float, a: float, b: float, gamma: float, kbt: float, power: float, stressPeriod: float) -> None

  
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
    

        """
        pass


