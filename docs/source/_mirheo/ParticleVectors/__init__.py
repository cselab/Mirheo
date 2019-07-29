class Mesh:
    r"""
        Internally used class for describing a simple triangular mesh
    
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
        In contrast with the simple :any:`Mesh`, this class precomputes some required quantities on the mesh, 
        including connectivity structures and stress-free quantities.        
    
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(off_filename: str) -> None


            Create a mesh by reading the OFF file.
            The stress free shape is the input initial mesh
            
            Args:
                off_filename: path of the OFF file
        

2. __init__(off_initial_mesh: str, off_stress_free_mesh: str) -> None


            Create a mesh by reading the OFF file, with a different stress free shape.
            
            Args:
                off_initial_mesh: path of the OFF file : initial mesh
                off_stress_free_mesh: path of the OFF file : stress-free mesh)
        

3. __init__(vertices: List[List[float[3]]], faces: List[List[int[3]]]) -> None


        Create a mesh by giving coordinates and connectivity
        
        Args:
            vertices: vertex coordinates
            faces:    connectivity: one triangle per entry, each integer corresponding to the vertex indices
        

4. __init__(vertices: List[List[float[3]]], stress_free_vertices: List[List[float[3]]], faces: List[List[int[3]]]) -> None


        Create a mesh by giving coordinates and connectivity, with a different stress-free shape.
        
        Args:
            vertices: vertex coordinates
            stress_free_vertices: vertex coordinates of the stress-free shape
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
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
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
                object_size: number of frozen particles per object
                mesh: :any:`Mesh` object used for bounce back and dump
        

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

class RodVector(ObjectVector):
    r"""
        Rod Vector is an :any:`ObjectVector` which reprents rod geometries.
    
    """
    def __init__():
        r"""__init__(name: str, mass: float, num_segments: int) -> None



            Args:
                name: name of the created Rod Vector
                mass: mass of a single particle
                num_segments: number of elements to discretize the rod
        

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

class RigidCapsuleVector(RigidObjectVector):
    r"""
        :any:`RigidObjectVector` specialized for capsule shapes.
        The advantage is that it doesn't need mesh and moment of inertia define, as those can be computed analytically.
    
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(name: str, mass: float, object_size: int, radius: float, length: float) -> None



            Args:
                name: name of the created PV
                mass: mass of a single particle
                object_size: number of frozen particles per object
                radius: radius of the capsule
                length: length of the capsule between the half balls. The total height is then "length + 2 * radius"


        

2. __init__(name: str, mass: float, object_size: int, radius: float, length: float, mesh: ParticleVectors.Mesh) -> None


            Args:
                name: name of the created PV
                mass: mass of a single particle
                object_size: number of frozen particles per object
                radius: radius of the capsule
                length: length of the capsule between the half balls. The total height is then "length + 2 * radius"
                mesh: :any:`Mesh` object representing the shape of the object. This is used for dump only.
        

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

class RigidCylinderVector(RigidObjectVector):
    r"""
        :any:`RigidObjectVector` specialized for cylindrical shapes.
        The advantage is that it doesn't need mesh and moment of inertia define, as those can be computed analytically.
    
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(name: str, mass: float, object_size: int, radius: float, length: float) -> None


            Args:
                name: name of the created PV
                mass: mass of a single particle
                object_size: number of frozen particles per object
                radius: radius of the cylinder
                length: length of the cylinder

        

2. __init__(name: str, mass: float, object_size: int, radius: float, length: float, mesh: ParticleVectors.Mesh) -> None


            Args:
                name: name of the created PV
                mass: mass of a single particle
                object_size: number of frozen particles per object
                radius: radius of the cylinder
                length: length of the cylinder
                mesh: :any:`Mesh` object representing the shape of the object. This is used for dump only.
        

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

class RigidEllipsoidVector(RigidObjectVector):
    r"""
        :any:`RigidObjectVector` specialized for ellipsoidal shapes.
        The advantage is that it doesn't need mesh and moment of inertia define, as those can be computed analytically.
    
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(name: str, mass: float, object_size: int, semi_axes: Tuple[float, float, float]) -> None



            Args:
                name: name of the created PV
                mass: mass of a single particle
                object_size: number of frozen particles per object
                semi_axes: ellipsoid principal semi-axes
        

2. __init__(name: str, mass: float, object_size: int, semi_axes: Tuple[float, float, float], mesh: ParticleVectors.Mesh) -> None



            Args:
                name: name of the created PV
                mass: mass of a single particle
                object_size: number of frozen particles per object
                radius: radius of the cylinder
                semi_axes: ellipsoid principal semi-axes
                mesh: :any:`Mesh` object representing the shape of the object. This is used for dump only.

        

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


