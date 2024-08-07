class PintUnitsConverter(object):
    r"""Converts given values to the given unit system.
    """
    def __init__():
        r"""None
        """
        pass

class ComQ:
    r"""None
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(arg0: float, arg1: float, arg2: float, arg3: float, arg4: float, arg5: float, arg6: float) -> None

2. __init__(arg0: real3, arg1: real4) -> None

3. __init__(arg0: List[float]) -> None

        """
        pass

class CudaArrayInterface:
    r"""
        Cupy and numba-compatible view for an internal CUDA buffer.
    
    """
    @property
    def __cuda_array_interface__():
        r"""
        The dictionary describing the underlying CUDA buffer.

        For more information, see:
            https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
        and
            https://docs.cupy.dev/en/stable/reference/interoperability.html#numba
    
        """
        pass

class DomainInfo:
    r"""
        Convert between local domain coordinates (specific to each rank) and global domain coordinates.
    
    """
    def global_to_local():
        r"""global_to_local(x: real3) -> real3


            Convert from global coordinates to local coordinates.

            Args:
                x: Position in global coordinates.
        

        """
        pass

    def is_in_subdomain():
        r"""is_in_subdomain(x: real3) -> bool


            Returns True if the given position (in global coordinates) is inside the subdomain of the current rank,
            False otherwise.

            Args:
                x: Position in global coordinates.
        

        """
        pass

    def local_to_global():
        r"""local_to_global(x: real3) -> real3


            Convert local coordinates to global coordinates.

            Args:
                x: Position in local coordinates.
        

        """
        pass

    @property
    def global_size():
        r"""
            Size of the whole simulation domain.
        
        """
        pass

    @property
    def global_start():
        r"""
            Subdomain lower corner position of the current rank, in global coordinates.
        
        """
        pass

    @property
    def global_to_local_shift():
        r"""
            shift to transform global coordinates to local coordinates.
        
        """
        pass

    @property
    def local_size():
        r"""
            Subdomain extents of the current rank.
        
        """
        pass

    @property
    def local_to_global_shift():
        r"""
            shift to transform local coordinates to global coordinates.
        
        """
        pass

class MirState:
    r"""
        state of the simulation shared by all simulation objects.
    
    """
    @property
    def current_dt():
        r"""
            Current simulation step size dt.
            Note: this property is accessible only while Mirheo::run() is running.
        
        """
        pass

    @property
    def current_step():
        r"""
            Current simulation step.
        
        """
        pass

    @property
    def current_time():
        r"""
            Current simulation time.
        
        """
        pass

    @property
    def domain_info():
        r"""
            The :any:`DomainInfo` of the current rank.
        
        """
        pass

class Mirheo:
    r"""
        Main coordination class, should only be one instance at a time
    
    """
    def __init__():
        r"""__init__(nranks: int3, domain: real3, log_filename: str='log', debug_level: int=-1, checkpoint_every: int=0, checkpoint_folder: str='restart/', checkpoint_mode: str='PingPong', max_obj_half_length: float=0.0, cuda_aware_mpi: bool=False, no_splash: bool=False, comm_ptr: int=0) -> None


Create the Mirheo coordinator.

.. warning::
    Debug level determines the amount of output produced by each of the simulation processes:

        0. silent: no log output
        1. only report fatal errors
        2. report serious errors
        3. report important information steps of simulation and warnings (this is the default level)
        4. report not critical information
        5. report some debug information
        6. report more debug
        7. report all the debug
        8. force flushing to the file after each message

    Debug levels above 4 or 5 may significanlty increase the runtime, they are only recommended to debug errors.
    Flushing increases the runtime yet more, but it is required in order not to lose any messages in case of abnormal program abort.

    The default debug level may be modified by setting the ``MIRHEO_DEBUG_LEVEL`` environment variable to the desired value.
    This variable may be useful when Mirheo is linked as part of other codes, in which case the ``debug_level`` variable affects only parts of the execution.


Args:
    nranks: number of MPI simulation tasks per axis: x,y,z. If postprocess is enabled, the same number of the postprocess tasks will be running
    domain: size of the simulation domain in x,y,z. Periodic boundary conditions are applied at the domain boundaries. The domain will be split in equal chunks between the MPI ranks.
        The largest chunk size that a single MPI rank can have depends on the total number of particles,
        handlers and hardware, and is typically about :math:`120^3 - 200^3`.
    log_filename: prefix of the log files that will be created.
        Logging is implemented in the form of one file per MPI rank, so in the simulation folder NP files with names log_00000.log, log_00001.log, ...
        will be created, where NP is the total number of MPI ranks.
        Each MPI task (including postprocess) writes messages about itself into his own log file, and the combined log may be created by merging all
        the individual ones and sorting with respect to time.
        If this parameter is set to 'stdout' or 'stderr' standard output or standard error streams will be used instead of the file, however,
        there is no guarantee that messages from different ranks are synchronized.
    debug_level: Debug level from 0 to 8, see above.
    checkpoint_every: save state of the simulation components (particle vectors and handlers like integrators, plugins, etc.)
    checkpoint_folder: folder where the checkpoint files will reside (for Checkpoint mechanism), or folder prefix (for Snapshot mechanism)
    checkpoint_mode: set to "PingPong" to keep only the last 2 checkpoint states; set to "Incremental" to keep all checkpoint states.
    max_obj_half_length: Half of the maximum size of all objects. Needs to be set when objects are self interacting with pairwise interactions.
    cuda_aware_mpi: enable CUDA Aware MPI. The MPI library must support that feature, otherwise it may fail.
    no_splash: don't display the splash screen when at the start-up.
    comm_ptr: pointer to communicator. By default MPI_COMM_WORLD will be used
        

        """
        pass

    def applyObjectBelongingChecker():
        r"""applyObjectBelongingChecker(checker: mirheo::ObjectBelongingChecker, pv: mirheo::ParticleVector, correct_every: int=0, inside: str='', outside: str='') -> mirheo::ParticleVector


                Apply the **checker** to the given particle vector.
                One and only one of the options **inside** or **outside** has to be specified.

                Args:
                    checker: instance of :any:`BelongingChecker`
                    pv: :any:`ParticleVector` that will be split (source PV)
                    inside: if specified and not "none", a new :any:`ParticleVector` with name **inside** will be returned, that will keep the inner particles of the source PV. If set to "none", None object will be returned. In any case, the source PV will only contain the outer particles
                    outside: if specified and not "none", a new :any:`ParticleVector` with name **outside** will be returned, that will keep the outer particles of the source PV. If set to "none", None object will be returned. In any case, the source PV will only contain the inner particles
                    correct_every: If greater than zero, perform correction every this many time-steps.
                        Correction will move e.g. *inner* particles of outer PV to the :inner PV
                        and viceversa. If one of the PVs was defined as "none", the 'wrong' particles will be just removed.

                Returns:
                    New :any:`ParticleVector` or None depending on **inside** and **outside** options

        

        """
        pass

    def computeVolumeInsideWalls():
        r"""computeVolumeInsideWalls(walls: List[mirheo::Wall], nSamplesPerRank: int=100000) -> float


                Compute the volume inside the given walls in the whole domain (negative values are the 'inside' of the simulation).
                The computation is made via simple Monte-Carlo.

                Args:
                    walls: sdf based walls
                    nSamplesPerRank: number of Monte-Carlo samples used per rank
        

        """
        pass

    def deregisterIntegrator():
        r"""deregisterIntegrator(integrator: mirheo::Integrator) -> None

Deregister a integrator.

        """
        pass

    def deregisterPlugins():
        r"""deregisterPlugins(arg0: mirheo::SimulationPlugin, arg1: mirheo::PostprocessPlugin) -> None

Deregister a plugin.

        """
        pass

    def dumpWalls2XDMF():
        r"""dumpWalls2XDMF(walls: List[mirheo::Wall], h: real3, filename: str='xdmf/wall') -> None


                Write Signed Distance Function for the intersection of the provided walls (negative values are the 'inside' of the simulation)

                Args:
                    walls: list of walls to dump; the output sdf will be the union of all walls inside
                    h: cell-size of the resulting grid
                    filename: base filename output, will create to files filename.xmf and filename.h5
        

        """
        pass

    def getState():
        r"""getState(self: Mirheo) -> MirState

Return mirheo state

        """
        pass

    def isComputeTask():
        r"""isComputeTask(self: Mirheo) -> bool

Returns ``True`` if the current rank is a simulation task and ``False`` if it is a postrprocess task

        """
        pass

    def isMasterTask():
        r"""isMasterTask(self: Mirheo) -> bool

Returns ``True`` if the current rank is the root

        """
        pass

    def log_compile_options():
        r"""log_compile_options(self: Mirheo) -> None


             output compile times options in the log
        

        """
        pass

    def makeFrozenRigidParticles():
        r"""makeFrozenRigidParticles(checker: mirheo::ObjectBelongingChecker, shape: mirheo::ObjectVector, icShape: mirheo::InitialConditions, interactions: List[mirheo::Interaction], integrator: mirheo::Integrator, number_density: float, mass: float=1.0, dt: float, nsteps: int=1000) -> mirheo::ParticleVector


                Create particles frozen inside object.

                .. note::
                    A separate simulation will be run for every call to this function, which may take certain amount of time.
                    If you want to save time, consider using restarting mechanism instead

                Args:
                    checker: object belonging checker
                    shape: object vector describing the shape of the rigid object
                    icShape: initial conditions for shape
                    interactions: list of :any:`Interaction` that will be used to construct the equilibrium particles distribution
                    integrator: this :any:`Integrator` will be used to construct the equilibrium particles distribution
                    number_density: target particle number density
                    mass: the mass of a single frozen particle
                    dt: time step
                    nsteps: run this many steps to achieve equilibrium

                Returns:
                    New :any:`ParticleVector` that will contain particles that are close to the wall boundary, but still inside the wall.

        

        """
        pass

    def makeFrozenWallParticles():
        r"""makeFrozenWallParticles(pvName: str, walls: List[mirheo::Wall], interactions: List[mirheo::Interaction], integrator: mirheo::Integrator, number_density: float, mass: float=1.0, dt: float, nsteps: int=1000) -> mirheo::ParticleVector


                Create particles frozen inside the walls.

                .. note::
                    A separate simulation will be run for every call to this function, which may take certain amount of time.
                    If you want to save time, consider using restarting mechanism instead

                Args:
                    pvName: name of the created particle vector
                    walls: array of instances of :any:`Wall` for which the frozen particles will be generated
                    interactions: list of :any:`Interaction` that will be used to construct the equilibrium particles distribution
                    integrator: this :any:`Integrator` will be used to construct the equilibrium particles distribution
                    number_density: target particle number density
                    mass: the mass of a single frozen particle
                    dt: time step
                    nsteps: run this many steps to achieve equilibrium

                Returns:
                    New :any:`ParticleVector` that will contain particles that are close to the wall boundary, but still inside the wall.

        

        """
        pass

    def registerBouncer():
        r"""registerBouncer(bouncer: mirheo::Bouncer) -> None


               Register Object Bouncer

               Args:
                   bouncer: the :any:`Bouncer` to register
        

        """
        pass

    def registerIntegrator():
        r"""registerIntegrator(integrator: mirheo::Integrator) -> None


                Register an :any:`Integrator` to the coordinator

                Args:
                    integrator: the :any:`Integrator` to register
         

        """
        pass

    def registerInteraction():
        r"""registerInteraction(interaction: mirheo::Interaction) -> None


                Register an :any:`Interaction` to the coordinator

                Args:
                    interaction: the :any:`Interaction` to register
        

        """
        pass

    def registerObjectBelongingChecker():
        r"""registerObjectBelongingChecker(checker: mirheo::ObjectBelongingChecker, ov: mirheo::ObjectVector) -> None


                Register Object Belonging Checker

                Args:
                    checker: instance of :any:`BelongingChecker`
                    ov: :any:`ObjectVector` belonging to which the **checker** will check
        

        """
        pass

    def registerParticleVector():
        r"""registerParticleVector(pv: mirheo::ParticleVector, ic: mirheo::InitialConditions=None) -> None


            Register particle vector

            Args:
                pv: :any:`ParticleVector`
                ic: :class:`~InitialConditions.InitialConditions` that will generate the initial distibution of the particles
        

        """
        pass

    def registerPlugins():
        r"""registerPlugins(arg0: mirheo::SimulationPlugin, arg1: mirheo::PostprocessPlugin) -> None

Register Plugins

        """
        pass

    def registerWall():
        r"""registerWall(wall: mirheo::Wall, check_every: int=0) -> None


               Register a :any:`Wall`.

               Args:
                   wall: the :any:`Wall` to register
                   check_every: if positive, check every this many time steps if particles penetrate the walls
        

        """
        pass

    def restart():
        r"""restart(folder: str='restart/') -> None


               Restart the simulation. This function should typically be called just before running the simulation.
               It will read the state of all previously registered instances of :any:`ParticleVector`, :any:`Interaction`, etc.
               If the folder contains no checkpoint file required for one of those, an error occur.

               .. warning::
                  Certain :any:`Plugins` may not implement restarting mechanism and will not restart correctly.
                  Please check the documentation for the plugins.

               Args:
                   folder: folder with the checkpoint files
        

        """
        pass

    def run():
        r"""run(niters: int, dt: float) -> None


             Advance the system for a given amount of time steps.

             Args:
                 niters: number of time steps to advance
                 dt: time step duration
        

        """
        pass

    def save_dependency_graph_graphml():
        r"""save_dependency_graph_graphml(fname: str, current: bool=True) -> None


             Exports `GraphML <http://graphml.graphdrawing.org/>`_ file with task graph for the current simulation time-step

             Args:
                 fname: the output filename (without extension)
                 current: if True, save the current non empty tasks; else, save all tasks that can exist in a simulation

             .. warning::
                 if current is set to True, this must be called **after** :py:meth:`mmirheo.Mirheo.run`.
         

        """
        pass

    def setBouncer():
        r"""setBouncer(bouncer: mirheo::Bouncer, ov: mirheo::ObjectVector, pv: mirheo::ParticleVector) -> None


                Assign a :any:`Bouncer` between an :any:`ObjectVector` and a :any:`ParticleVector`.

                Args:
                    bouncer: :any:`Bouncer` compatible with the object vector
                    ov: the :any:`ObjectVector` to be bounced on
                    pv: the :any:`ParticleVector` to be bounced
        

        """
        pass

    def setIntegrator():
        r"""setIntegrator(integrator: mirheo::Integrator, pv: mirheo::ParticleVector) -> None


               Set a specific :any:`Integrator` to a given :any:`ParticleVector`

               Args:
                   integrator: the :any:`Integrator` to assign
                   pv: the concerned :any:`ParticleVector`
        

        """
        pass

    def setInteraction():
        r"""setInteraction(interaction: mirheo::Interaction, pv1: mirheo::ParticleVector, pv2: mirheo::ParticleVector) -> None


                Forces between two instances of :any:`ParticleVector` (they can be the same) will be computed according to the defined interaction.

                Args:
                    interaction: :any:`Interaction` to apply
                    pv1: first :any:`ParticleVector`
                    pv2: second :any:`ParticleVector`

        

        """
        pass

    def setWall():
        r"""setWall(wall: mirheo::Wall, pv: mirheo::ParticleVector, maximum_part_travel: float=0.25) -> None


                Assign a :any:`Wall` bouncer to a given :any:`ParticleVector`.
                The current implementation does not support :any:`ObjectVector`.

                Args:
                    wall: the :any:`Wall` surface which will bounce the particles
                    pv: the :any:`ParticleVector` to be bounced
                    maximum_part_travel: maximum distance that one particle travels in one time step.
                        this should be as small as possible for performance reasons but large enough for correctness
         

        """
        pass

    def start_profiler():
        r"""start_profiler(self: Mirheo) -> None

Tells nvprof to start recording timeline

        """
        pass

    def stop_profiler():
        r"""stop_profiler(self: Mirheo) -> None

Tells nvprof to stop recording timeline

        """
        pass

class int2:
    r"""None
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(arg0: int, arg1: int) -> None

2. __init__(arg0: List[int]) -> None

        """
        pass

    @property
    def x():
        r"""
        """
        pass

    @property
    def y():
        r"""
        """
        pass

class int3:
    r"""None
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(arg0: int, arg1: int, arg2: int) -> None

2. __init__(arg0: List[int]) -> None

        """
        pass

    @property
    def x():
        r"""
        """
        pass

    @property
    def y():
        r"""
        """
        pass

    @property
    def z():
        r"""
        """
        pass

class real2:
    r"""None
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(arg0: float, arg1: float) -> None

2. __init__(arg0: List[float]) -> None

        """
        pass

    @property
    def x():
        r"""
        """
        pass

    @property
    def y():
        r"""
        """
        pass

class real3:
    r"""None
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(arg0: float, arg1: float, arg2: float) -> None

2. __init__(arg0: List[float]) -> None

        """
        pass

    @property
    def x():
        r"""
        """
        pass

    @property
    def y():
        r"""
        """
        pass

    @property
    def z():
        r"""
        """
        pass

class real4:
    r"""None
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(arg0: float, arg1: float, arg2: float, arg3: float) -> None

2. __init__(arg0: List[float]) -> None

        """
        pass

    @property
    def w():
        r"""
        """
        pass

    @property
    def x():
        r"""
        """
        pass

    @property
    def y():
        r"""
        """
        pass

    @property
    def z():
        r"""
        """
        pass



__all__ = ["tools", "version"]
from .version import mir_version
