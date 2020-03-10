class Integrator:
    r"""
        Base integration class
    
    """
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
        """
        pass

class Minimize(Integrator):
    r"""
            Energy minimization integrator. Updates particle positions according to a gradient-descent policy with respect to the energy potential (force).
            Does not read or modify particle velocities.

            .. math::

                \mathbf{a}^{n} &= \frac{1}{m} \mathbf{F}(\mathbf{x}^{n}, \mathbf{v}^{n-1/2}) \\
                \mathbf{x}^{n+1} &= \mathbf{x}^{n} + \frac{\Delta t^2}{m} \mathbf{a}^n
        
    """
    def __init__():
        r"""__init__(name: str, max_displacement: float) -> None


                Args:
                    name: name of the integrator
                    max_displacement: maximum displacement per time step
            

        """
        pass

class Oscillate(Integrator):
    r"""
        Move particles with the periodically changing velocity
        :math:`\mathbf{u}(t) = \cos(2 \pi \, t / T) \mathbf{u}_0`
    
    """
    def __init__():
        r"""__init__(name: str, velocity: real3, period: float) -> None


                Args:
                    name: name of the integrator
                    velocity: :math:`\mathbf{u}_0`
                    period: oscillation period :math:`T`
            

        """
        pass

class RigidVelocityVerlet(Integrator):
    r"""
        Integrate the position and rotation (in terms of quaternions) of the rigid bodies as per Velocity-Verlet scheme.
        Can only applied to :any:`RigidObjectVector` or :any:`RigidEllipsoidVector`.
    
    """
    def __init__():
        r"""__init__(name: str) -> None


                Args:
                    name: name of the integrator
            

        """
        pass

class Rotate(Integrator):
    r"""
        Rotate particles around the specified point in space with a constant angular velocity :math:`\mathbf{\Omega}`
    
    """
    def __init__():
        r"""__init__(name: str, center: real3, omega: real3) -> None


                Args:
                    name: name of the integrator
                    center: point around which to rotate
                    omega: angular velocity :math:`\mathbf{\Omega}`
            

        """
        pass

class SubStep(Integrator):
    r"""
            Takes advantage of separation of time scales between "fast" internal forces and other "slow" forces on an object vector.
            This integrator advances the object vector with constant slow forces for 'substeps' sub time steps.
            The fast forces are updated after each sub step.
            Positions and velocity are updated using an internal velocity verlet integrator.
        
    """
    def __init__():
        r"""__init__(name: str, substeps: int, fastForces: List[Interactions.Interaction]) -> None


                Args:
                    name: name of the integrator
                    substeps: number of sub steps
                    fastForces: a list of fast interactions. Only accepts :any:`MembraneForces` or :any:`RodForces`
                
                .. warning::
                    The interaction will be set to the required object vector when setting this integrator to the object vector.
                    Hence the interaction needs not to be set explicitely to the OV.
            

        """
        pass

class Translate(Integrator):
    r"""
        Translate particles with a constant velocity :math:`\mathbf{u}` regardless forces acting on them.
    
    """
    def __init__():
        r"""__init__(name: str, velocity: real3) -> None


                Args:
                    name: name of the integrator
                    velocity: translational velocity :math:`\mathbf{\Omega}`
            

        """
        pass

class VelocityVerlet(Integrator):
    r"""
            Classical Velocity-Verlet integrator with fused steps for coordinates and velocities.
            The velocities are shifted with respect to the coordinates by one half of the time-step
            
            .. math::

                \mathbf{a}^{n} &= \frac{1}{m} \mathbf{F}(\mathbf{x}^{n}, \mathbf{v}^{n-1/2}) \\
                \mathbf{v}^{n+1/2} &= \mathbf{v}^{n-1/2} + \mathbf{a}^n \Delta t \\
                \mathbf{x}^{n+1} &= \mathbf{x}^{n} + \mathbf{v}^{n+1/2} \Delta t 

            where bold symbol means a vector, :math:`m` is a particle mass, and superscripts denote the time: :math:`\mathbf{x}^{k} = \mathbf{x}(k \, \Delta t)`
        
    """
    def __init__():
        r"""__init__(name: str) -> None


                Args:
                    name: name of the integrator
            

        """
        pass

class VelocityVerlet_withConstForce(Integrator):
    r"""
            Same as regular :any:`VelocityVerlet`, but the forces on all the particles are modified with the constant pressure term:
   
            .. math::

                \mathbf{a}^{n} &= \frac{1}{m} \left( \mathbf{F}(\mathbf{x}^{n}, \mathbf{v}^{n-1/2}) + \mathbf{F}_{extra} \right) \\
        
    """
    def __init__():
        r"""__init__(name: str, force: real3) -> None



                Args:
                    name: name of the integrator
                    force: :math:`\mathbf{F}_{extra}`
            

        """
        pass

class VelocityVerlet_withPeriodicForce(Integrator):
    r"""
            Same as regular Velocity-Verlet, but the forces on all the particles are modified with periodic Poiseuille term.
            This means that all the particles in half domain along certain axis (Ox, Oy or Oz) are pushed with force
            :math:`F_{Poiseuille}` parallel to Oy, Oz or Ox correspondingly, and the particles in another half of the domain are pushed in the same direction
            with force :math:`-F_{Poiseuille}`    
        
    """
    def __init__():
        r"""__init__(name: str, force: float, direction: str) -> None

                
                Args:
                    name: name of the integrator
                    force: force magnitude, :math:`F_{Poiseuille}`
                    direction: Valid values: \"x\", \"y\", \"z\". Defines the direction of the pushing force
                               if direction is \"x\", the sign changes along \"y\".
                               if direction is \"y\", the sign changes along \"z\".
                               if direction is \"z\", the sign changes along \"x\".
            

        """
        pass


