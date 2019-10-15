class Bouncer:
    r"""
        Base class for bouncing particles off the objects.
        Take bounce kernel as argument:
        
        * **kernel** = "bounce_back":
            bounces back the particle.

        * **kernel** = "bounce_maxwell":
            reinsert particle at the collision point with a velocity drawn from a maxwellian distribution.
    
    """
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
        """
        pass

class Capsule(Bouncer):
    r"""
        This bouncer will use the analytical capsule representation of the rigid objects to perform the bounce.
        No additional correction from the Object Belonging Checker is usually required.
        The velocity of the particles bounced from the cylinder is reversed with respect to the boundary velocity at the contact point.
    
    """
    def __init__():
        r"""__init__(name: str, kernel: str, **kwargs) -> None


            Args:
                name: name of the checker
                kernel: the kernel used to bounce the particles (see :any:`Bouncer`)
            
        

        """
        pass

class Cylinder(Bouncer):
    r"""
        This bouncer will use the analytical cylinder representation of the rigid objects to perform the bounce.
        No additional correction from the Object Belonging Checker is usually required.
        The velocity of the particles bounced from the cylinder is reversed with respect to the boundary velocity at the contact point.
    
    """
    def __init__():
        r"""__init__(name: str, kernel: str, **kwargs) -> None


            Args:
                name: name of the checker
                kernel: the kernel used to bounce the particles (see :any:`Bouncer`)
            
        

        """
        pass

class Ellipsoid(Bouncer):
    r"""
        This bouncer will use the analytical ellipsoid representation of the rigid objects to perform the bounce.
        No additional correction from the Object Belonging Checker is usually required.
        The velocity of the particles bounced from the ellipsoid is reversed with respect to the boundary velocity at the contact point.
    
    """
    def __init__():
        r"""__init__(name: str, kernel: str, **kwargs) -> None


            Args:
                name: name of the checker
                kernel: the kernel used to bounce the particles (see :any:`Bouncer`)
            
        

        """
        pass

class Mesh(Bouncer):
    r"""
        This bouncer will use the triangular mesh associated with objects to detect boundary crossings.
        Therefore it can only be created for Membrane and Rigid Object types of object vectors.
        Due to numerical precision, about :math:`1` of :math:`10^5 - 10^6` mesh crossings will not be detected, therefore it is advised to use that bouncer in
        conjunction with correction option provided by the Object Belonging Checker, see :ref:`user-belongers`.
        
        .. note::
            In order to prevent numerical instabilities in case of light membrane particles,
            the new velocity of the bounced particles will be a random vector drawn from the Maxwell distibution of given temperature
            and added to the velocity of the mesh triangle at the collision point.
    
    """
    def __init__():
        r"""__init__(name: str, kernel: str, **kwargs) -> None


            Args:
                name: name of the bouncer
                kernel: the kernel used to bounce the particles (see :any:`Bouncer`)
        

        """
        pass

class Rod(Bouncer):
    r"""
        This bouncer will use the analytical representation of enlarged segments by a given radius.
        The velocity of the particles bounced from the segments is reversed with respect to the boundary velocity at the contact point.
    
    """
    def __init__():
        r"""__init__(name: str, radius: float, kernel: str, **kwargs) -> None


            Args:
                name: name of the checker
                radius: radius of the segments
                kernel: the kernel used to bounce the particles (see :any:`Bouncer`)
            
        

        """
        pass


