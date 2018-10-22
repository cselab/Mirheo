class udevicex:
    r"""
        Main coordination class, should only be one instance at a time
    
    """
    def __init__():
        r"""__init__(*args, **kwargs)
Overloaded function.

1. __init__(nranks: Tuple[int, int, int], domain: Tuple[float, float, float], log_filename: str = 'log', debug_level: int = 3, checkpoint_every: int = 0, restart_folder: str = 'restart/', cuda_aware_mpi: bool = False, no_splash: bool = False) -> None


            Args:
                nranks:
                    number of MPI simulation tasks per axis: x,y,z. If postprocess is enabled, the same number of the postprocess tasks will be running
                domain:
                    size of the simulation domain in x,y,z. Periodic boundary conditions are applied at the domain boundaries.
                    The domain will be split in equal chunks between the MPI ranks.
                    The largest chunk size that a single MPI rank can have depends on the total number of particles,
                    handlers and hardware, and is typically about :math:`120^3 - 200^3`.
                log_filename:
                    prefix of the log files that will be created. 
                    Logging is implemented in the form of one file per MPI rank, so in the simulation folder NP files with names log_00000.log, log_00001.log, ... will be created, where NP is the total number of MPI ranks. 
                    Each MPI task (including postprocess) writes messages about itself into his own log file, and the combined log may be created by merging all the individual ones and sorting with respect to time.
                    If this parameter is set to 'stdout' or 'stderr' standard output or standard error streams will be used instead of the file, however, there is no guarantee that messages from different ranks are synchronized
                debug_level:
                    Debug level varies from 1 to 8:
                    
                       #. only report fatal errors
                       #. report serious errors
                       #. report warnings (this is the default level)
                       #. report not critical information
                       #. report some debug information
                       #. report more debug
                       #. report all the debug
                       #. force flushing to the file after each message
                    
                    Debug levels above 4 or 5 may significanlty increase the runtime, they are only recommended to debug errors.
                    Flushing increases the runtime yet more, but it is required in order not to lose any messages in case of abnormal program abort.
                checkpoint_every:
                    save state of the simulation components (particle vectors and handlers like integrators, plugins, etc.)
                restart_folder:
                    folder where the checkpoint files will reside
                cuda_aware_mpi: enable CUDA Aware MPI (GPU RDMA). As of now it may crash, or may yield slower execution.
                no_splash: Don't display the splash screen when at the start-up.
                
        

2. __init__(commPtr: int, nranks: Tuple[int, int, int], domain: Tuple[float, float, float], log_filename: str = 'log', debug_level: int = 3, checkpoint_every: int = 0, restart_folder: str = 'restart/', cuda_aware_mpi: bool = False) -> None


            Args:
                commPtr: pointer to communicator
        

        """
        pass

    def applyObjectBelongingChecker():
        r"""applyObjectBelongingChecker(checker: ObjectBelongingChecker, pv: ParticleVector, correct_every: int = 0, inside: str = '', outside: str = '') -> ParticleVector


                Apply the **checker** to the given particle vector.
                One and only one of the options **inside** or **outside** has to be specified.
                
                Args:
                    checker: instance of :any:`BelongingChecker`
                    pv: :any:`ParticleVector` that will be split (source PV) 
                    inside:
                        if specified and not "none", a new :any:`ParticleVector` with name **inside** will be returned, that will keep the inner particles of the source PV. If set to "none", None object will be returned. In any case, the source PV will only contain the outer particles
                    outside:
                        if specified and not "none", a new :any:`ParticleVector` with name **outside** will be returned, that will keep the outer particles of the source PV. If set to "none", None object will be returned. In any case, the source PV will only contain the inner particles
                    correct_every:
                        If greater than zero, perform correction every this many time-steps.                        
                        Correction will move e.g. *inner* particles of outer PV to the :inner PV
                        and viceversa. If one of the PVs was defined as "none", the 'wrong' particles will be just removed.
                            
                Returns:
                    New :any:`ParticleVector` or None depending on **inside** and **outside** options
                    
        

        """
        pass

    def dumpWalls2XDMF():
        r"""dumpWalls2XDMF(walls: List[Wall], h: Tuple[float, float, float], filename: str = 'xdmf/wall') -> None


                Write Signed Distance Function for the intersection of the provided walls (negative values are the 'inside' of the simulation)
                
                Args:
                    h: cell-size of the resulting grid                    
        

        """
        pass

    def isComputeTask():
        r"""isComputeTask(self: udevicex) -> bool

Returns whether current rank will do compute or postrprocess

        """
        pass

    def isMasterTask():
        r"""isMasterTask(self: udevicex) -> bool

Returns whether current task is the very first one

        """
        pass

    def makeFrozenRigidParticles():
        r"""makeFrozenRigidParticles(checker: ObjectBelongingChecker, shape: ObjectVector, icShape: InitialConditions, interaction: Interaction, integrator: Integrator, density: float, nsteps: int = 1000) -> ParticleVector


                Create particles frozen inside object.
                
                .. note::
                    A separate simulation will be run for every call to this function, which may take certain amount of time.
                    If you want to save time, consider using restarting mechanism instead
                
                Args:
                    checker: object belonging checker
                    shape: object vector describing the shape of the rigid object
                    icShape: initial conditions for shape
                    interaction: this :any:`Interaction` will be used to construct the equilibrium particles distribution
                    integrator: this :any:`Integrator` will be used to construct the equilibrium particles distribution
                    density: target particle density
                    nsteps: run this many steps to achieve equilibrium
                            
                Returns:
                    New :any:`ParticleVector` that will contain particles that are close to the wall boundary, but still inside the wall.
                    
        

        """
        pass

    def makeFrozenWallParticles():
        r"""makeFrozenWallParticles(pvName: str, walls: List[Wall], interaction: Interaction, integrator: Integrator, density: float, nsteps: int = 1000) -> ParticleVector


                Create particles frozen inside the walls.
                
                .. note::
                    A separate simulation will be run for every call to this function, which may take certain amount of time.
                    If you want to save time, consider using restarting mechanism instead
                
                Args:
                    pvName: name of the created particle vector
                    walls: array of instances of :any:`Wall` for which the frozen particles will be generated
                    interaction: this :any:`Interaction` will be used to construct the equilibrium particles distribution
                    integrator: this :any:`Integrator` will be used to construct the equilibrium particles distribution
                    density: target particle density
                    nsteps: run this many steps to achieve equilibrium
                            
                Returns:
                    New :any:`ParticleVector` that will contain particles that are close to the wall boundary, but still inside the wall.
                    
        

        """
        pass

    def registerBouncer():
        r"""registerBouncer(arg0: Bouncer) -> None

Register Object Bouncer

        """
        pass

    def registerIntegrator():
        r"""registerIntegrator(arg0: Integrator) -> None

Register Integrator

        """
        pass

    def registerInteraction():
        r"""registerInteraction(arg0: Interaction) -> None

Register Interaction

        """
        pass

    def registerObjectBelongingChecker():
        r"""registerObjectBelongingChecker(checker: ObjectBelongingChecker, ov: ObjectVector) -> None


                Register Object Belonging Checker
                
                Args:
                    checker: instance of :any:`BelongingChecker`
                    ov: :any:`ObjectVector` belonging to which the **checker** will check
        

        """
        pass

    def registerParticleVector():
        r"""registerParticleVector(pv: ParticleVector, ic: InitialConditions = None, checkpoint_every: int = 0) -> None


            Register particle vector
            
            Args:
                pv: :any:`ParticleVector`
                ic: :class:`~InitialConditions.InitialConditions` that will generate the initial distibution of the particles
                checkpoint_every:
                    every that many timesteps the state of the Particle Vector across all the MPI processes will be saved to disk  into the ./restart/ folder. The checkpoint files may be used to restart the whole simulation or only some individual PVs from the saved states. Default value of 0 means no checkpoint.
        

        """
        pass

    def registerPlugins():
        r"""registerPlugins(arg0: SimulationPlugin, arg1: PostprocessPlugin) -> None

Register Plugins

        """
        pass

    def registerWall():
        r"""registerWall(wall: Wall, check_every: int = 0) -> None

Register Wall

        """
        pass

    def run():
        r"""run(arg0: int) -> None

Run the simulation

        """
        pass

    def save_dependency_graph_graphml():
        r"""save_dependency_graph_graphml(arg0: str) -> None


            Exports `GraphML <http://graphml.graphdrawing.org/>`_ file with task graph for the current simulation time-step

        """
        pass

    def setBouncer():
        r"""setBouncer(arg0: Bouncer, arg1: ObjectVector, arg2: ParticleVector) -> None

Set Bouncer

        """
        pass

    def setIntegrator():
        r"""setIntegrator(arg0: Integrator, arg1: ParticleVector) -> None

Set Integrator

        """
        pass

    def setInteraction():
        r"""setInteraction(interaction: Interaction, pv1: ParticleVector, pv2: ParticleVector) -> None


                Forces between two Particle Vectors (they can be the same) *name1* and *name2* will be computed according to the defined interaction.
        

        """
        pass

    def setWall():
        r"""setWall(arg0: Wall, arg1: ParticleVector) -> None

Set Wall

        """
        pass

    def start_profiler():
        r"""start_profiler(self: udevicex) -> None

Tells nvprof to start recording timeline

        """
        pass

    def stop_profiler():
        r"""stop_profiler(self: udevicex) -> None

Tells nvprof to stop recording timeline

        """
        pass


