class InitialConditions:
    r"""
            Base class for initial conditions
        
    """
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
        """
        pass

class FromArray(InitialConditions):
    r"""
        Set particles according to given position and velocity arrays.
    
    """
    def __init__():
        r"""__init__(pos: List[List[float[3]]], vel: List[List[float[3]]]) -> None


            Args:
                pos: array of positions
                vel: array of velocities
        

        """
        pass

class Membrane(InitialConditions):
    r"""
        Can only be used with Membrane Object Vector, see :ref:`user-ic`. These IC will initialize the particles of each object
        according to the mesh associated with Membrane, and then the objects will be translated/rotated according to the provided initial conditions.
    
    """
    def __init__():
        r"""__init__(com_q: List[List[float[7]]], global_scale: float = 1.0) -> None


            Args:
                com_q:
                    List describing location and rotation of the created objects.               
                    One entry in the list corresponds to one object created.                          
                    Each entry consist of 7 floats: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where    
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized 
                global_scale:
                    All the membranes will be scaled by that value. Useful to implement membranes growth so that they
                    can fill the space with high volume fraction                                        
        

        """
        pass

class Restart(InitialConditions):
    r"""
        Read the state of the particle vector from restart files.
    
    """
    def __init__():
        r"""__init__(path: str = 'restart/') -> None



            Args:
                path: folder where the restart files reside.
        

        """
        pass

class Rigid(InitialConditions):
    r"""
        Can only be used with Rigid Object Vector or Rigid Ellipsoid, see :ref:`user-ic`. These IC will initialize the particles of each object
        according to the template .xyz file and then the objects will be translated/rotated according to the provided initial conditions.
            
    
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(com_q: List[List[float[7]]], xyz_filename: str) -> None


            Args:
                com_q:
                    List describing location and rotation of the created objects.               
                    One entry in the list corresponds to one object created.                          
                    Each entry consist of 7 floats: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where    
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized 
                xyz_filename:
                    Template that describes the positions of the body particles before translation or        
                    rotation is applied. Standard .xyz file format is used with first line being             
                    the number of particles, second comment, third and onwards - particle coordinates.       
                    The number of particles in the file must be the same as in number of particles per object
                    in the corresponding PV
        

2. __init__(com_q: List[List[float[7]]], coords: List[List[float[3]]]) -> None


            Args:
                com_q:
                    List describing location and rotation of the created objects.               
                    One entry in the list corresponds to one object created.                          
                    Each entry consist of 7 floats: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where    
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized 
                coords:
                    Template that describes the positions of the body particles before translation or        
                    rotation is applied.       
                    The number of coordinates must be the same as in number of particles per object
                    in the corresponding PV
        

3. __init__(com_q: List[List[float[7]]], coords: List[List[float[3]]], init_vels: List[List[float[3]]]) -> None


            Args:
                com_q:
                    List describing location and rotation of the created objects.               
                    One entry in the list corresponds to one object created.                          
                    Each entry consist of 7 floats: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where    
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized 
                coords:
                    Template that describes the positions of the body particles before translation or        
                    rotation is applied.       
                    The number of coordinates must be the same as in number of particles per object
                    in the corresponding PV
                com_q:
                    List specifying initial Center-Of-Mass velocities of the bodies.               
                    One entry (list of 3 floats) in the list corresponds to one object 
        

        """
        pass

class Rod(InitialConditions):
    r"""
        Can only be used with Rod Vector. These IC will initialize the particles of each rod
        according to the the given explicit center-line position aand torsion mapping and then 
        the objects will be translated/rotated according to the provided initial conditions.
            
    
    """
    def __init__():
        r"""__init__(com_q: List[List[float[7]]], center_line: Callable[[float], Tuple[float, float, float]], torsion: Callable[[float], float], a: float, initial_frame: Tuple[float, float, float] = (inf, inf, inf)) -> None


            Args:
                com_q:
                    List describing location and rotation of the created objects.               
                    One entry in the list corresponds to one object created.                          
                    Each entry consist of 7 floats: *<com_x> <com_y> <com_z>  <q_x> <q_y> <q_z> <q_w>*, where    
                    *com* is the center of mass of the object, *q* is the quaternion of its rotation,
                    not necessarily normalized 
                center_line:
                    explicit mapping :math:`\mathbf{r} : [0,1] \rightarrow R^3`. 
                    Assume :math:`|r'(s)|` is constant for all :math:`s \in [0,1]`.
                torsion:
                    explicit mapping :math:`\tau : [0,1] \rightarrow R`.
                a:
                    width of the rod 
                initial_frame:
                    Orientation of the initial frame (optional)
                    By default, will come up with any orthogonal frame to the rod at origin
        

        """
        pass

class Uniform(InitialConditions):
    r"""
        The particles will be generated with the desired number density uniformly at random in all the domain.
        These IC may be used with any Particle Vector, but only make sense for regular PV.
            
    
    """
    def __init__():
        r"""__init__(density: float) -> None


            Args:
                density: target density
        

        """
        pass

class UniformFiltered(InitialConditions):
    r"""
        The particles will be generated with the desired number density uniformly at random in all the domain and then filtered out by the given filter.
        These IC may be used with any Particle Vector, but only make sense for regular PV.            
    
    """
    def __init__():
        r"""__init__(density: float, filter: Callable[[Tuple[float, float, float]], bool]) -> None


            Args:
                density: target density
                filter: given position, returns True if the particle should be kept 
        

        """
        pass

class UniformSphere(InitialConditions):
    r"""
        The particles will be generated with the desired number density uniformly at random inside or outside a given sphere.
        These IC may be used with any Particle Vector, but only make sense for regular PV.
            
    
    """
    def __init__():
        r"""__init__(density: float, center: Tuple[float, float, float], radius: float, inside: bool) -> None


            Args:
                density: target density
                center: center of the sphere
                radius: radius of the sphere
                inside: whether the particles should be inside or outside the sphere
        

        """
        pass


