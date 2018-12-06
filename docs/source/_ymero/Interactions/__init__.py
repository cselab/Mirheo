class Interaction:
    r"""Base interaction class
    """
class JuelicherBendingParameters:
    r"""
        Bending parameters for Juelicher model
    
    """
    def __init__():
        r"""__init__(self: Interactions.JuelicherBendingParameters) -> None


            Structure keeping parameters of the bending membrane interaction
        

        """
        pass

class KantorBendingParameters:
    r"""
        Bending parameters for Kantor model
    
    """
    def __init__():
        r"""__init__(self: Interactions.KantorBendingParameters) -> None


            Structure keeping parameters of the bending membrane interaction
        

        """
        pass

class MembraneParameters:
    r"""
        Common membrane parameters
    
    """
    def __init__():
        r"""__init__(self: Interactions.MembraneParameters) -> None


            Structure keeping parameters of the membrane interaction
        

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
            J. Chem. Phys., 107(11), 4423–4435. `doi <https://doi.org/10.1063/1.474784>`_
    
    """
    def __init__():
        r"""__init__(name: str, rc: float, a: float, gamma: float, kbt: float, dt: float, power: float) -> None

  
            Args:
            name: name of the interaction
                rc: interaction cut-off (no forces between particles further than **rc** apart)
                a: :math:`a`
                gamma: :math:`\gamma`
                kbt: :math:`k_B T`
                dt: time-step, that for consistency has to be the same as the integration time-step for the corresponding particle vectors
                power: :math:`p` in the weight function
    

        """
        pass

    def setSpecificPair():
        r"""setSpecificPair(pv1: ParticleVectors.ParticleVector, pv2: ParticleVectors.ParticleVector, a: float = inf, gamma: float = inf, kbt: float = inf, dt: float = inf, power: float = inf) -> None


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

    
    """
class MembraneForcesJuelicher(Interaction):
    r"""
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
    
    """
    def __init__():
        r"""__init__(name: str, params: Interactions.MembraneParameters, params_bending: Interactions.JuelicherBendingParameters, stressFree: bool, grow_until: float = 0) -> None

 
             Args:
                 name: name of the interaction
                 params: instance of :any: `MembraneParameters`
                 params_bending: instance of :any: `JuelicherBendingParameters`
                 stressFree: equilibrium bond length and areas are taken from the initial mesh
                 grow_until: time to grow the cell at initialization stage; 
                             the size increases linearly in time from half of the provided mesh to its full size after that time
                             the parameters are scaled accordingly with time
    

        """
        pass

class MembraneForcesKantor(Interaction):
    r"""
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

    
    """
    def __init__():
        r"""__init__(name: str, params: Interactions.MembraneParameters, params_bending: Interactions.KantorBendingParameters, stressFree: bool, grow_until: float = 0) -> None

 
             Args:
                 name: name of the interaction
                 params: instance of :any: `MembraneParameters`
                 params_bending: instance of :any: `KantorBendingParameters`
                 stressFree: equilibrium bond length and areas are taken from the initial mesh
                 grow_until: time to grow the cell at initialization stage; 
                             the size increases linearly in time from half of the provided mesh to its full size after that time
                             the parameters are scaled accordingly with time
    

        """
        pass

class DPDWithStress(DPD):
    r"""
        wrapper of :any:`DPD` with, in addition, stress computation
    
    """
    def __init__():
        r"""__init__(name: str, stressName: str, rc: float, a: float, gamma: float, kbt: float, dt: float, power: float, stressPeriod: float) -> None

  
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
    

        """
        pass

    def setSpecificPair():
        r"""setSpecificPair(pv1: ParticleVectors.ParticleVector, pv2: ParticleVectors.ParticleVector, a: float = inf, gamma: float = inf, kbt: float = inf, dt: float = inf, power: float = inf) -> None


            Override some of the interaction parameters for a specific pair of Particle Vectors
         

        """
        pass

class LJWithStress(LJ):
    r"""
        wrapper of :any:`LJ` with, in addition, stress computation
    
    """
    def __init__():
        r"""__init__(name: str, stressName: str, rc: float, epsilon: float, sigma: float, max_force: float = 1000.0, object_aware: bool, stressPeriod: float) -> None


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
    

        """
        pass

    def setSpecificPair():
        r"""setSpecificPair(pv1: ParticleVectors.ParticleVector, pv2: ParticleVectors.ParticleVector, epsilon: float, sigma: float, max_force: float) -> None


            Override some of the interaction parameters for a specific pair of Particle Vectors
        

        """
        pass


