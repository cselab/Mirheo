class Mesh:
    r"""
        Internally used class for desctibing a simple triangular mesh
    
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(off_filename: str) -> None


        Create a mesh by reading the OFF file
        Args:
            off_filename: path of the OFF file
    

2. __init__(vertices: List[List[float[3]]], faces: List[List[int[3]]]) -> None


        Create a mesh by giving coordinates and connectivity
        Args:
            vertices: vertex coordinates
            faces:    connectivity: one triangle per entry, each integer corresponding to the vertex indices
    

        """
        pass

    def getTriangles():
        r"""getTriangles(self: ParticleVectors.Mesh) -> List[List[int[3]]]


        returns the vertex indices for each triangle of the mesh.
    

        """
        pass

    def getVertices():
        r"""getVertices(self: ParticleVectors.Mesh) -> List[List[float[3]]]


        returns the vertex coordinates of the mesh.
    

        """
        pass

class ParticleVector:
    r"""
        Basic particle vector, consists of identical disconnected particles.
    
    """
    def __init__():
        r"""__init__(name: str, mass: float) -> None


            Args:
                name: name of the created PV 
                mass: mass of a single particle
        

        """
        pass

    def getCoordinates():
        r"""getCoordinates(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[List[float[3]]]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[List[float[3]]]) -> None


            Args:
                forces: A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[List[float[3]]]) -> None


            Args:
                velocities: A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        

        """
        pass

class MembraneMesh(Mesh):
    r"""
        Internally used class for desctibing a triangular mesh that can be used with the Membrane Interactions.
        In contrast with the simple :any:`Mesh`, this class precomputes some required quantities on the mesh
    
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(off_filename: str) -> None


            Create a mesh by reading the OFF file
            Args:
                off_filename: path of the OFF file
        

2. __init__(vertices: List[List[float[3]]], faces: List[List[int[3]]]) -> None


        Create a mesh by giving coordinates and connectivity
        Args:
            vertices: vertex coordinates
            faces:    connectivity: one triangle per entry, each integer corresponding to the vertex indices
    

        """
        pass

    def getTriangles():
        r"""getTriangles(self: ParticleVectors.Mesh) -> List[List[int[3]]]


        returns the vertex indices for each triangle of the mesh.
    

        """
        pass

    def getVertices():
        r"""getVertices(self: ParticleVectors.Mesh) -> List[List[float[3]]]


        returns the vertex coordinates of the mesh.
    

        """
        pass

class ObjectVector(ParticleVector):
    r"""
        Basic Object Vector
    
    """
    def getCoordinates():
        r"""getCoordinates(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[List[float[3]]]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[List[float[3]]]) -> None


            Args:
                forces: A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[List[float[3]]]) -> None


            Args:
                velocities: A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        

        """
        pass

class MembraneVector(ObjectVector):
    r"""
        Membrane is an Object Vector representing cell membranes.
        It must have a triangular mesh associated with it such that each particle is mapped directly onto single mesh vertex.
    
    """
    def __init__():
        r"""__init__(name: str, mass: float, mesh: ParticleVectors.MembraneMesh) -> None


            Args:
                name: name of the created PV 
                mass: mass of a single particle
                mesh: :any:`MembraneMesh` object                
        

        """
        pass

    def getCoordinates():
        r"""getCoordinates(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[List[float[3]]]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[List[float[3]]]) -> None


            Args:
                forces: A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[List[float[3]]]) -> None


            Args:
                velocities: A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        

        """
        pass

class RigidEllipsoidVector(ObjectVector):
    r"""
        Rigid Ellipsoid is the same as the Rigid Object except that it can only represent ellipsoidal shapes.
        The advantage is that it doesn't need mesh and moment of inertia define, as those can be computed analytically.
    
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(name: str, mass: float, object_size: int, semi_axes: Tuple[float, float, float]) -> None


                Args:
                    name: name of the created PV 
                    mass: mass of a single particle
                    object_size: number of particles per membrane, must be the same as the number of vertices of the mesh
                    semi_axes: ellipsoid principal semi-axes
        

2. __init__(name: str, mass: float, object_size: int, semi_axes: Tuple[float, float, float], mesh: ParticleVectors.Mesh) -> None


                Args:
                    name: name of the created PV 
                    mass: mass of a single particle
                    object_size: number of particles per membrane, must be the same as the number of vertices of the mesh
                    semi_axes: ellipsoid principal semi-axes
                    mesh: mesh representing the shape of the ellipsoid. This is used for dump only.
        

        """
        pass

    def getCoordinates():
        r"""getCoordinates(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[List[float[3]]]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[List[float[3]]]) -> None


            Args:
                forces: A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[List[float[3]]]) -> None


            Args:
                velocities: A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        

        """
        pass

class RigidObjectVector(ObjectVector):
    r"""
        Rigid Object is an Object Vector representing objects that move as rigid bodies, with no relative displacement against each other in an object.
        It must have a triangular mesh associated with it that defines the shape of the object.
    
    """
    def __init__():
        r"""__init__(name: str, mass: float, inertia: Tuple[float, float, float], object_size: int, mesh: ParticleVectors.Mesh) -> None

 
                Args:
                    name: name of the created PV 
                    mass: mass of a single particle
                    inertia: moment of inertia of the body in its principal axes. The principal axes of the mesh are assumed to be aligned with the default global *OXYZ* axes
                    object_size: number of particles per membrane, must be the same as the number of vertices of the mesh
                    mesh: :any:`MembraneMesh` object         
        

        """
        pass

    def getCoordinates():
        r"""getCoordinates(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns: 
                A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[List[float[3]]]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` floats: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[List[float[3]]]) -> None


            Args:
                forces: A list of :math:`N \times 3` floats: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[List[float[3]]]) -> None


            Args:
                velocities: A list of :math:`N \times 3` floats: 3 components of velocity for every of the N particles
        

        """
        pass


