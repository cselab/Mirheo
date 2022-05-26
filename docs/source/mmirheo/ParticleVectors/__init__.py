class DataManager:
    r"""
        A collection of channels in pinned memory.
    
    """
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
        """
        pass

class LocalParticleVector:
    r"""
        Particle local data storage, composed of particle channels.
    
    """
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
        """
        pass

    @property
    def per_particle():
        r"""
            The :any:`DataManager` that contains the particle channels.
        
        """
        pass

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
    

2. __init__(vertices: List[real3], faces: List[int3]) -> None


        Create a mesh by giving coordinates and connectivity

        Args:
            vertices: vertex coordinates
            faces:    connectivity: one triangle per entry, each integer corresponding to the vertex indices

    

        """
        pass

    def getFaces():
        r"""getFaces(self: ParticleVectors.Mesh) -> List[List[int[3]]]


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
                A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[real3]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[real3]) -> None


            Args:
                forces: A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[real3]) -> None


            Args:
                velocities: A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    @property
    def halo():
        r"""
            The halo LocalParticleVector instance, the storage of halo particles.
        
        """
        pass

    @property
    def local():
        r"""
            The local LocalParticleVector instance, the storage of local particles.
        
        """
        pass

class LocalObjectVector(LocalParticleVector):
    r"""
        Object vector local data storage, additionally contains object channels.
    
    """
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
        """
        pass

    @property
    def per_object():
        r"""
            The :any:`DataManager` that contains the object channels.
        
        """
        pass

    @property
    def per_particle():
        r"""
            The :any:`DataManager` that contains the particle channels.
        
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
        

3. __init__(vertices: List[real3], faces: List[int3]) -> None


            Create a mesh by giving coordinates and connectivity

            Args:
                vertices: vertex coordinates
                faces:    connectivity: one triangle per entry, each integer corresponding to the vertex indices
        

4. __init__(vertices: List[real3], stress_free_vertices: List[real3], faces: List[int3]) -> None


            Create a mesh by giving coordinates and connectivity, with a different stress-free shape.

            Args:
                vertices: vertex coordinates
                stress_free_vertices: vertex coordinates of the stress-free shape
                faces:    connectivity: one triangle per entry, each integer corresponding to the vertex indices
    

        """
        pass

    def getFaces():
        r"""getFaces(self: ParticleVectors.Mesh) -> List[List[int[3]]]


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
        Basic Object Vector.
        An Object Vector stores chunks of particles, each chunk belonging to the same object.

        .. warning::
            In case of interactions with other :any:`ParticleVector`, the extents of the objects must be smaller than a subdomain size. The code only issues a run time warning but it is the responsibility of the user to ensure this condition for correctness.

    
    """
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
        """
        pass

    def getCoordinates():
        r"""getCoordinates(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[real3]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[real3]) -> None


            Args:
                forces: A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[real3]) -> None


            Args:
                velocities: A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    @property
    def halo():
        r"""
            The halo LocalObjectVector instance, the storage of halo objects.
        
        """
        pass

    @property
    def local():
        r"""
            The local LocalObjectVector instance, the storage of local objects.
        
        """
        pass

class ChainVector(ObjectVector):
    r"""
        Object Vector representing chain of particles.
    
    """
    def __init__():
        r"""__init__(name: str, mass: float, chain_length: int) -> None


            Args:
                name: name of the created PV
                mass: mass of a single particle
                chain_length: number of particles per chain
        

        """
        pass

    def getCoordinates():
        r"""getCoordinates(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[real3]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[real3]) -> None


            Args:
                forces: A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[real3]) -> None


            Args:
                velocities: A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    @property
    def halo():
        r"""
            The halo LocalObjectVector instance, the storage of halo objects.
        
        """
        pass

    @property
    def local():
        r"""
            The local LocalObjectVector instance, the storage of local objects.
        
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
                A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[real3]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[real3]) -> None


            Args:
                forces: A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[real3]) -> None


            Args:
                velocities: A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    @property
    def halo():
        r"""
            The halo LocalObjectVector instance, the storage of halo objects.
        
        """
        pass

    @property
    def local():
        r"""
            The local LocalObjectVector instance, the storage of local objects.
        
        """
        pass

class RigidObjectVector(ObjectVector):
    r"""
        Rigid Object is an Object Vector representing objects that move as rigid bodies, with no relative displacement against each other in an object.
        It must have a triangular mesh associated with it that defines the shape of the object.
    
    """
    def __init__():
        r"""__init__(name: str, mass: float, inertia: real3, object_size: int, mesh: ParticleVectors.Mesh) -> None



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
                A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[real3]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[real3]) -> None


            Args:
                forces: A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[real3]) -> None


            Args:
                velocities: A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    @property
    def halo():
        r"""
            The halo LocalObjectVector instance, the storage of halo objects.
        
        """
        pass

    @property
    def local():
        r"""
            The local LocalObjectVector instance, the storage of local objects.
        
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
                A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[real3]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[real3]) -> None


            Args:
                forces: A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[real3]) -> None


            Args:
                velocities: A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    @property
    def halo():
        r"""
            The halo LocalObjectVector instance, the storage of halo objects.
        
        """
        pass

    @property
    def local():
        r"""
            The local LocalObjectVector instance, the storage of local objects.
        
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
                A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[real3]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[real3]) -> None


            Args:
                forces: A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[real3]) -> None


            Args:
                velocities: A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    @property
    def halo():
        r"""
            The halo LocalObjectVector instance, the storage of halo objects.
        
        """
        pass

    @property
    def local():
        r"""
            The local LocalObjectVector instance, the storage of local objects.
        
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
                A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[real3]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[real3]) -> None


            Args:
                forces: A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[real3]) -> None


            Args:
                velocities: A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    @property
    def halo():
        r"""
            The halo LocalObjectVector instance, the storage of halo objects.
        
        """
        pass

    @property
    def local():
        r"""
            The local LocalObjectVector instance, the storage of local objects.
        
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

1. __init__(name: str, mass: float, object_size: int, semi_axes: real3) -> None



            Args:
                name: name of the created PV
                mass: mass of a single particle
                object_size: number of frozen particles per object
                semi_axes: ellipsoid principal semi-axes
        

2. __init__(name: str, mass: float, object_size: int, semi_axes: real3, mesh: ParticleVectors.Mesh) -> None



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
                A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def getForces():
        r"""getForces(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def getVelocities():
        r"""getVelocities(self: ParticleVectors.ParticleVector) -> List[List[float[3]]]


            Returns:
                A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    def get_indices():
        r"""get_indices(self: ParticleVectors.ParticleVector) -> List[int]


            Returns:
                A list of unique integer particle identifiers
        

        """
        pass

    def setCoordinates():
        r"""setCoordinates(coordinates: List[real3]) -> None


            Args:
                coordinates: A list of :math:`N \times 3` reals: 3 components of coordinate for every of the N particles
        

        """
        pass

    def setForces():
        r"""setForces(forces: List[real3]) -> None


            Args:
                forces: A list of :math:`N \times 3` reals: 3 components of force for every of the N particles
        

        """
        pass

    def setVelocities():
        r"""setVelocities(velocities: List[real3]) -> None


            Args:
                velocities: A list of :math:`N \times 3` reals: 3 components of velocity for every of the N particles
        

        """
        pass

    @property
    def halo():
        r"""
            The halo LocalObjectVector instance, the storage of halo objects.
        
        """
        pass

    @property
    def local():
        r"""
            The local LocalObjectVector instance, the storage of local objects.
        
        """
        pass


# Functions

def getReservedBisegmentChannels():
    r"""getReservedBisegmentChannels() -> List[str]

Return the list of reserved channel names per bisegment fields

    """
    pass

def getReservedObjectChannels():
    r"""getReservedObjectChannels() -> List[str]

Return the list of reserved channel names for object fields

    """
    pass

def getReservedParticleChannels():
    r"""getReservedParticleChannels() -> List[str]

Return the list of reserved channel names for particle fields

    """
    pass


