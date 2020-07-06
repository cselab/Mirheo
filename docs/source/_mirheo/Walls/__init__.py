class Wall:
    r"""
        Base wall class.
    
    """
    def __init__():
        r"""Initialize self.  See help(type(self)) for accurate signature.
        """
        pass

    def attachFrozenParticles():
        r"""attachFrozenParticles(arg0: ParticleVectors.ParticleVector) -> None


        Let the wall know that the following :any:`ParticleVector` should be treated as frozen.
        As a result, its particles will not be removed from the inside of the wall.
    

        """
        pass

class Box(Wall):
    r"""
        Rectangular cuboid wall with edges aligned with the coordinate axes.

    
    """
    def __init__():
        r"""__init__(name: str, low: real3, high: real3, inside: bool = False) -> None


            Args:
                name: name of the wall
                low: lower corner of the box
                high: higher corner of the box
                inside: whether the domain is inside the box or outside of it
        

        """
        pass

    def attachFrozenParticles():
        r"""attachFrozenParticles(arg0: ParticleVectors.ParticleVector) -> None


        Let the wall know that the following :any:`ParticleVector` should be treated as frozen.
        As a result, its particles will not be removed from the inside of the wall.
    

        """
        pass

class Cylinder(Wall):
    r"""
        Cylindrical infinitely stretching wall, the main axis is aligned along OX or OY or OZ
    
    """
    def __init__():
        r"""__init__(name: str, center: real2, radius: float, axis: str, inside: bool = False) -> None


            Args:
                name: name of the wall
                center: point that belongs to the cylinder axis projected along that axis
                radius: cylinder radius
                axis: direction of cylinder axis, valid values are "x", "y" or "z"
                inside: whether the domain is inside the cylinder or outside of it
        

        """
        pass

    def attachFrozenParticles():
        r"""attachFrozenParticles(arg0: ParticleVectors.ParticleVector) -> None


        Let the wall know that the following :any:`ParticleVector` should be treated as frozen.
        As a result, its particles will not be removed from the inside of the wall.
    

        """
        pass

class MovingPlane(Wall):
    r"""
        Planar wall that is moving along itself with constant velocity.
        Can be used to produce Couette velocity profile in combination with
        The boundary conditions on such wall are no-through and constant velocity (specified).
    
    """
    def __init__():
        r"""__init__(name: str, normal: real3, pointThrough: real3, velocity: real3) -> None


            Args:
                name: name of the wall
                normal: wall normal, pointing *inside* the wall
                pointThrough: point that belongs to the plane
                velocity: wall velocity, should be orthogonal to the normal
        

        """
        pass

    def attachFrozenParticles():
        r"""attachFrozenParticles(arg0: ParticleVectors.ParticleVector) -> None


        Let the wall know that the following :any:`ParticleVector` should be treated as frozen.
        As a result, its particles will not be removed from the inside of the wall.
    

        """
        pass

class OscillatingPlane(Wall):
    r"""
        Planar wall that is moving along itself with periodically changing velocity:

        .. math::
            \mathbf{u}(t) = cos(2*\pi * t / T);
    
    """
    def __init__():
        r"""__init__(name: str, normal: real3, pointThrough: real3, velocity: real3, period: float) -> None


            Args:
                name: name of the wall
                normal: wall normal, pointing *inside* the wall
                pointThrough: point that belongs to the plane
                velocity: velocity amplitude, should be orthogonal to the normal
                period: oscillation period dpd time units
        

        """
        pass

    def attachFrozenParticles():
        r"""attachFrozenParticles(arg0: ParticleVectors.ParticleVector) -> None


        Let the wall know that the following :any:`ParticleVector` should be treated as frozen.
        As a result, its particles will not be removed from the inside of the wall.
    

        """
        pass

class Plane(Wall):
    r"""
        Planar infinitely stretching wall. Inside is determined by the normal direction .

    
    """
    def __init__():
        r"""__init__(name: str, normal: real3, pointThrough: real3) -> None


            Args:
                name: name of the wall
                normal: wall normal, pointing *inside* the wall
                pointThrough: point that belongs to the plane
        

        """
        pass

    def attachFrozenParticles():
        r"""attachFrozenParticles(arg0: ParticleVectors.ParticleVector) -> None


        Let the wall know that the following :any:`ParticleVector` should be treated as frozen.
        As a result, its particles will not be removed from the inside of the wall.
    

        """
        pass

class RotatingCylinder(Wall):
    r"""
        Cylindrical wall rotating with constant angular velocity along its axis.
    
    """
    def __init__():
        r"""__init__(name: str, center: real2, radius: float, axis: str, omega: float, inside: bool = False) -> None


            Args:
                name: name of the wall
                center: point that belongs to the cylinder axis projected along that axis
                radius: cylinder radius
                axis: direction of cylinder axis, valid values are "x", "y" or "z"
                omega: angular velocity of rotation along the cylinder axis
                inside: whether the domain is inside the cylinder or outside of it
        

        """
        pass

    def attachFrozenParticles():
        r"""attachFrozenParticles(arg0: ParticleVectors.ParticleVector) -> None


        Let the wall know that the following :any:`ParticleVector` should be treated as frozen.
        As a result, its particles will not be removed from the inside of the wall.
    

        """
        pass

class SDF(Wall):
    r"""
        This wall is based on an arbitrary Signed Distance Function (SDF) defined in the simulation domain on a regular Cartesian grid.
        The wall reads the SDF data from a custom format ``.sdf`` file, that has a special structure.

        First two lines define the header: three real number separated by spaces govern the size of the domain where the SDF is defined,
        and next three integer numbers (:math:`Nx\,\,Ny\,\,Nz`) define the resolution.
        Next the :math:`Nx \times Ny \times Nz` single precision realing point values are written (in binary representation).

        Negative SDF values correspond to the domain, and positive -- to the inside of the wall.
        The boundary is defined by the zero-level isosurface.
    
    """
    def __init__():
        r"""__init__(name: str, sdfFilename: str, h: real3 = real3(0.25, 0.25, 0.25)) -> None


            Args:
                name: name of the wall
                sdfFilename: name of the ``.sdf`` file
                h: resolution of the resampled SDF.
                   In order to have a more accurate SDF representation, the initial function is resampled on a finer grid.
                   The lower this value is, the more accurate the wall will be represented, however, the  more memory it will consume and the slower the execution will be.
        

        """
        pass

    def attachFrozenParticles():
        r"""attachFrozenParticles(arg0: ParticleVectors.ParticleVector) -> None


        Let the wall know that the following :any:`ParticleVector` should be treated as frozen.
        As a result, its particles will not be removed from the inside of the wall.
    

        """
        pass

class Sphere(Wall):
    r"""
        Spherical wall.

    
    """
    def __init__():
        r"""__init__(name: str, center: real3, radius: float, inside: bool = False) -> None


            Args:
                name: name of the wall
                center: sphere center
                radius: sphere radius
                inside: whether the domain is inside the sphere or outside of it
        

        """
        pass

    def attachFrozenParticles():
        r"""attachFrozenParticles(arg0: ParticleVectors.ParticleVector) -> None


        Let the wall know that the following :any:`ParticleVector` should be treated as frozen.
        As a result, its particles will not be removed from the inside of the wall.
    

        """
        pass


