class Interaction:
    r"""Base interaction class
    """
class MembraneParameters:
    r"""None
    """
    def __init__():
        r"""__init__(self: Interactions.MembraneParameters) -> None


            Structure keeping parameters of the membrane interaction
        

        """
        pass

class DPD(Interaction):
    r"""
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
        Pairwise interaction according to the classical Lennard-Jones potential `http://rspa.royalsocietypublishing.org/content/106/738/463`
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
        Mesh-based forces acting on a membrane according to the model in PUT LINK
    
    """
    def __init__():
        r"""__init__(name: str, params: Interactions.MembraneParameters, stressFree: bool, grow_until: float = 0) -> None

 TODO
        

        """
        pass


