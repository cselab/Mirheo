class BelongingChecker:
    r"""
        Base class for checking if particles belong to objects
    
    """
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
        """
        pass

class Capsule(BelongingChecker):
    r"""
        This checker will use the analytical representation of the capsule to detect *inside*-*outside* status.
    
    """
    def __init__():
        r"""__init__(name: str) -> None


            Args:
                name: name of the checker
            

        """
        pass

class Cylinder(BelongingChecker):
    r"""
        This checker will use the analytical representation of the cylinder to detect *inside*-*outside* status.
    
    """
    def __init__():
        r"""__init__(name: str) -> None


            Args:
                name: name of the checker
            

        """
        pass

class Ellipsoid(BelongingChecker):
    r"""
        This checker will use the analytical representation of the ellipsoid to detect *inside*-*outside* status.
    
    """
    def __init__():
        r"""__init__(name: str) -> None


            Args:
                name: name of the checker
            

        """
        pass

class Mesh(BelongingChecker):
    r"""
        This checker will use the triangular mesh associated with objects to detect *inside*-*outside* status.
   
        .. note:
            Checking if particles are inside or outside the mesh is a computationally expensive task,
            so it's best to perform checks at most every 1'000 - 10'000 time-steps.
    
    """
    def __init__():
        r"""__init__(name: str) -> None


            Args:
                name: name of the checker
        

        """
        pass

class Rod(BelongingChecker):
    r"""
        This checker will detect *inside*-*outside* status with respect to every segment of the rod, enlarged by a given radius.
    
    """
    def __init__():
        r"""__init__(name: str, radius: float) -> None


            Args:
                name: name of the checker
                radius: radius of the rod
            

        """
        pass


