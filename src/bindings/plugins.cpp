#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <plugins/factory.h>
#include <core/xdmf/channel.h>

#include "bindings.h"
#include "class_wrapper.h"

using namespace pybind11::literals;

void exportPlugins(py::module& m)
{
    py::handlers_class<SimulationPlugin>  pysim(m, "SimulationPlugin", R"(
        Base simulation plugin class
    )");
    
    py::handlers_class<PostprocessPlugin> pypost(m, "PostprocessPlugin", R"(
        Base postprocess plugin class
    )");


    
    py::handlers_class<AddForcePlugin>(m, "AddForce", pysim, R"(
        This plugin will add constant force :math:`\mathbf{F}_{extra}` to each particle of a specific PV every time-step.
        Is is advised to only use it with rigid objects, since Velocity-Verlet integrator with constant pressure can do the same without any performance penalty.
    )");

    
    py::handlers_class<AddTorquePlugin>(m, "AddTorque", pysim, R"(
        This plugin will add constant torque :math:`\mathbf{T}_{extra}` to each *object* of a specific OV every time-step.
    )");

    py::handlers_class<AnchorParticlesPlugin>(m, "AnchorParticles", pysim, R"(
        This plugin will set a given particle at a given position and velocity.
    )");

    py::handlers_class<AnchorParticlesStatsPlugin>(m, "AnchorParticlesStats", pypost, R"(
        Postprocessing side of :any:`AnchorParticles` responsible to dump the data.
    )");

    py::handlers_class<Average3D>(m, "Average3D", pysim, R"(
        This plugin will project certain quantities of the particle vectors on the grid (by simple binning),
        perform time-averaging of the grid and dump it in XDMF (LINK) format with HDF5 (LINK) backend.
        The quantities of interest are represented as *channels* associated with particles vectors.
        Some interactions, integrators, etc. and more notable plug-ins can add to the Particle Vectors per-particles arrays to hold different values.
        These arrays are called *channels*.
        Any such channel may be used in this plug-in, however, user must explicitely specify the type of values that the channel holds.
        Particle number density is used to correctly average the values, so it will be sampled and written in any case.
        
        .. note::
            This plugin is inactive if postprocess is disabled
    )");

    py::handlers_class<AverageRelative3D>(m, "AverageRelative3D", pysim, R"(
        This plugin acts just like the regular flow dumper, with one difference.
        It will assume a coordinate system attached to the center of mass of a specific object.
        In other words, velocities and coordinates sampled correspond to the object reference frame.
        
        .. note::
            Note that this plugin needs to allocate memory for the grid in the full domain, not only in the corresponding MPI subdomain.
            Therefore large domains will lead to running out of memory
            
        .. note::
            This plugin is inactive if postprocess is disabled
    )");

    py::handlers_class<UniformCartesianDumper>(m, "UniformCartesianDumper", pypost, R"(
        Postprocess side plugin of :any:`Average3D` or :any:`AverageRelative3D`.
        Responsible for performing the I/O.
    )")
        .def("get_channel_view", [] (const UniformCartesianDumper &dumper, std::string chname) {
                auto ch = dumper.getChannelOrDie(chname);
                auto resolution = dumper.getLocalResolution();
                resolution.push_back(ch.nComponents());
            
                pybind11::dtype dt;
                if (ch.numberType == XDMF::Channel::NumberType::Float) dt = pybind11::dtype::of<float>();
                if (ch.numberType == XDMF::Channel::NumberType::Int)   dt = pybind11::dtype::of<int>();
            
                return py::array(dt, resolution, (float*)ch.data, py::cast(dumper));
            });

    py::handlers_class<DensityControlPlugin>(m, "DensityControlPlugin", pysim, R"(
        This plugin applies forces to a set of particle vectors in order to get a constant density.
    )");

    py::handlers_class<PostprocessDensityControl>(m, "PostprocessDensityControl", pypost, R"(
        Dumps info from :any:`DensityControlPlugin`.
    )");

    py::handlers_class<DensityOutletPlugin>(m, "DensityOutletPlugin", pysim, R"(
        This plugin removes particles from a set of :any:`ParticleVector` in a given region if the number density is larger than a given target.
    )");

    py::handlers_class<PlaneOutletPlugin>(m, "PlaneOutletPlugin", pysim, R"(
        This plugin removes all particles from a set of :any:`ParticleVector` that are on the non-negative side of a given plane.
    )");

    py::handlers_class<RateOutletPlugin>(m, "RateOutletPlugin", pysim, R"(
        This plugin removes particles from a set of :any:`ParticleVector` in a given region at a given mass rate.
    )");
    
    py::handlers_class<ExchangePVSFluxPlanePlugin>(m, "ExchangePVSFluxPlane", pysim, R"(
        This plugin exchanges particles from a particle vector crossing a given plane to another particle vector.
        A particle with position x, y, z has crossed the plane if ax + by + cz + d >= 0, where a, b, c and d are the coefficient 
        stored in the 'plane' variable
    )");

    py::handlers_class<ForceSaverPlugin>(m, "ForceSaver", pysim, R"(
        This plugin creates an extra channel per particle inside the given particle vector named 'forces'.
        It copies the total forces at each time step and make it accessible by other plugins.
        The forces are stored in an array of float3.
    )");
    
    py::handlers_class<ImposeProfilePlugin>(m, "ImposeProfile", pysim, R"(
        This plugin will set the velocity of each particle inside a given domain to a target velocity with an additive term 
        drawn from Maxwell distribution of the given temperature. 
    )");

    
    py::handlers_class<ImposeVelocityPlugin>(m, "ImposeVelocity", pysim, R"(
        This plugin will add velocity to all the particles of the target PV in the specified area (rectangle) such that the average velocity equals to desired.
        )")
        .def("set_target_velocity", &ImposeVelocityPlugin::setTargetVelocity);

    
    py::handlers_class<MagneticOrientationPlugin>(m, "MagneticOrientation", pysim, R"(
        This plugin gives a magnetic moment :math:`\mathbf{M}` to every rigid objects in a given :any:`RigidObjectVector`.
        It also models a uniform magnetic field :math:`\mathbf{B}` (varying in time) and adds the induced torque to the objects according to:

        .. math::

            \mathbf{T} = \mathbf{M} \times \mathbf{B}   

        The magnetic field is passed as a function from python.
        The function must take a float (time) as input and output a tuple of three floats (magnetic field).
    )");

    py::handlers_class<MembraneExtraForcePlugin>(m, "MembraneExtraForce", pysim, R"(
        This plugin adds a given external force to a given membrane. 
        The force is defined vertex wise and does not depend on position.
        It is the same for all membranes belonging to the same particle vector.
    )");

    
    py::handlers_class<MeshPlugin>(m, "MeshPlugin", pysim, R"(
        This plugin will write the meshes of all the object of the specified Object Vector in a PLY format (LINK).
   
        .. note::
            This plugin is inactive if postprocess is disabled
    )");

    py::handlers_class<MeshDumper>(m, "MeshDumper", pypost, R"(
        Postprocess side plugin of :any:`MeshPlugin`.
        Responsible for performing the data reductions and I/O.
    )");

    
    py::handlers_class<ObjStatsPlugin>(m, "ObjStats", pysim, R"(
        This plugin will write the coordinates of the centers of mass of the objects of the specified Object Vector.
        Instantaneous quantities (COM velocity, angular velocity, force, torque) are also written.
        If the objects are rigid bodies, also will be written the quaternion describing the rotation.
        
        The file format is the following:
        
        <object id> <simulation time> <COM>x3 [<quaternion>x4] <velocity>x3 <angular velocity>x3 <force>x3 <torque>x3
        
        .. note::
            Note that all the written values are *instantaneous*
            
        .. note::
            This plugin is inactive if postprocess is disabled
    )");

    py::handlers_class<ObjStatsDumper>(m, "ObjStatsDumper", pypost, R"(
        Postprocess side plugin of :any:`ObjStats`.
        Responsible for performing the I/O.
    )");

    py::handlers_class<ObjectPortalDestination>(m, "ObjectPortalDestination", pysim, R"(
        This plugin receives object vector content from external code.
    )");

    py::handlers_class<ObjectPortalSource>(m, "ObjectPortalSource", pysim, R"(
        This plugin sends object vector content to external code.
    )");

    py::handlers_class<ObjectToParticlesPlugin>(m, "ObjectToParticlesPlugin", pysim, R"(
        This plugin transforms objects to particles when they cross a given plane.
    )");

    py::handlers_class<ParticleChannelSaverPlugin>(m, "ParticleChannelSaver", pysim, R"(
        This plugin creates an extra channel per particle inside the given particle vector with a given name.
        It copies the content of an extra channel of pv at each time step and make it accessible by other plugins.
    )");

    py::handlers_class<ParticleCheckerPlugin>(m, "ParticleChecker", pysim, R"(
        This plugin will check the positions and velocities of all particles in the simulation every given time steps.
        To be used for debugging purpose.
    )");

    py::handlers_class<ParticleDragPlugin>(m, "ParticleDrag", pysim, R"(
        This plugin will add drag force :math:`\mathbf{f} = - C_d \mathbf{u}` to each particle of a specific PV every time-step.
    )");

    py::handlers_class<ParticlePortalDestination>(m, "ParticlePortalDestination", pysim, R"(
        This plugin receives particle vector content from external code.
    )");

    py::handlers_class<ParticlePortalSource>(m, "ParticlePortalSource", pysim, R"(
        This plugin sends particle vector content to external code.
    )");

    
    py::handlers_class<ParticleSenderPlugin>(m, "ParticleSenderPlugin", pysim, R"(
        This plugin will dump positions, velocities and optional attached data of all the particles of the specified Particle Vector.
        The data is dumped into hdf5 format. An additional xdfm file is dumped to describe the data and make it readable by visualization tools. 
    )");

    py::handlers_class<ParticleWithRodQuantitiesSenderPlugin>(m, "ParticleSenderWithRodDataPlugin", pysim, R"(
        Extension of :any:`ParticleSenderPlugin` to support bisegment data.
        If a field of optional data is per bisegment data (for a rod) this plugin will first scatter this data to particles.
    )");


    py::handlers_class<ParticleDisplacementPlugin>(m, "ParticleDisplacementPlugin", pysim, R"(
        This plugin computes and save the displacement of the particles within a given particle vector.
        The result is stored inside the extra channel "displacements" as an array of float3.
    )");

    
    py::handlers_class<ParticleDumperPlugin>(m, "ParticleDumperPlugin", pypost, R"(
        Postprocess side plugin of :any:`ParticleSenderPlugin`.
        Responsible for performing the I/O.
    )");

    
    py::handlers_class<ParticleWithMeshSenderPlugin>(m, "ParticleWithMeshSenderPlugin", pysim, R"(
        This plugin will dump positions, velocities and optional attached data of all the particles of the specified Object Vector, as well as connectivity information.
        The data is dumped into hdf5 format. An additional xdfm file is dumped to describe the data and make it readable by visualization tools. 
    )");

    py::handlers_class<ParticleWithMeshDumperPlugin>(m, "ParticleWithMeshDumperPlugin", pypost, R"(
        Postprocess side plugin of :any:`ParticleWithMeshSenderPlugin`.
        Responsible for performing the I/O.
    )");

    
    py::handlers_class<PinObjectPlugin>(m, "PinObject", pysim, R"(
        This plugin will impose given velocity as the center of mass velocity (by axis) of all the objects of the specified Object Vector.
        If the objects are rigid bodies, rotatation may be restricted with this plugin as well.
        The *time-averaged* force and/or torque required to impose the velocities and rotations are reported.
            
        .. note::
            This plugin is inactive if postprocess is disabled
    )").def_property_readonly_static("Unrestricted", [](py::object) { return PinObjectPlugin::Unrestricted; }, R"(
        Unrestricted
    )");

    py::handlers_class<PinRodExtremityPlugin>(m, "PinRodExtremity", pysim, R"(
        This plugin adds a force on a given segment of all the rods in a :any:`RodVector`.
        The force has the form deriving from the potential

        .. math::
            
            E = k \left( 1 - \cos \theta \right),

        where :math:`\theta` is the angle between the material frame and a given direction (projected on the concerned segment).
        Note that the force is applied only on the material frame and not on the center line.
    )");

    py::handlers_class<ReportPinObjectPlugin>(m, "ReportPinObject", pypost, R"(
        Postprocess side plugin of :any:`PinObject`.
        Responsible for performing the I/O.
    )");
    

    py::handlers_class<SimulationRadialVelocityControl>(m, "RadialVelocityControl", pysim, R"(
        This plugin applies a radial force (decreasing as :math:`r^3`) to all the particles of the target PVS.
        The force is adapted via a PID controller such that the average of the velocity times radial position of the particles matches a target value.
    )");

    py::handlers_class<PostprocessRadialVelocityControl>(m, "PostprocessRadialVelocityControl", pypost, R"(
        Postprocess side plugin of :any:`RadialVelocityControl`.
        Responsible for performing the I/O.
    )");


    py::handlers_class<SimulationStats>(m, "SimulationStats", pysim, R"(
        This plugin will report aggregate quantities of all the particles in the simulation:
        total number of particles in the simulation, average temperature and momentum, maximum velocity magnutide of a particle
        and also the mean real time per step in milliseconds.
        
        .. note::
            This plugin is inactive if postprocess is disabled
    )");

    py::handlers_class<PostprocessStats>(m, "PostprocessStats", pypost, R"(
        Postprocess side plugin of :any:`SimulationStats`.
        Responsible for performing the data reductions and I/O.
    )");

    
    py::handlers_class<SimulationVelocityControl>(m, "VelocityControl", pysim, R"(
        This plugin applies a uniform force to all the particles of the target PVS in the specified area (rectangle).
        The force is adapted bvia a PID controller such that the velocity average of the particles matches the target average velocity.
    )");

    py::handlers_class<PostprocessVelocityControl>(m, "PostprocessVelocityControl", pypost, R"(
        Postprocess side plugin of :any:`VelocityControl`.
        Responsible for performing the I/O.
    )");

    
    py::handlers_class<TemperaturizePlugin>(m, "Temperaturize", pysim, R"(
        This plugin changes the velocity of each particles from a given :any:`ParticleVector`.
        It can operate under two modes: `keepVelocity = True`, in which case it adds a term drawn from a Maxwell distribution to the current velocity;
        `keepVelocity = False`, in which case it sets the velocity to a term drawn from a Maxwell distribution.
    )");


    py::handlers_class<VelocityInletPlugin>(m, "VelocityInlet", pysim, R"(
        This plugin inserts particles in a given :any:`ParticleVector`.
        The particles are inserted on a given surface with given velocity inlet. 
        The rate of insertion is governed by the velocity and the given number density.
    )");
    
    
    py::handlers_class<VirialPressurePlugin>(m, "VirialPressure", pysim, R"(
        This plugin compute the virial pressure from a given :any:`ParticleVector`.
        Note that the stress computation must be enabled with the corresponding stressName.
        This returns the total internal virial part only (no temperature term).
        Note that the volume is not devided in the result, the user is responsible to properly scale the output.
    )");

    py::handlers_class<VirialPressureDumper>(m, "VirialPressureDumper", pypost, R"(
        Postprocess side plugin of :any:`VirialPressure`.
        Responsible for performing the I/O.
    )");
    
    py::handlers_class<WallRepulsionPlugin>(m, "WallRepulsion", pysim, R"(
        This plugin will add force on all the particles that are nearby a specified wall. The motivation of this plugin is as follows.
        The particles of regular PVs are prevented from penetrating into the walls by Wall Bouncers.
        However, using Wall Bouncers with Object Vectors may be undesirable (e.g. in case of a very viscous membrane) or impossible (in case of rigid objects).
        In these cases one can use either strong repulsive potential between the object and the wall particle or alternatively this plugin.
        The advantage of the SDF-based repulsion is that small penetrations won't break the simulation.
        
        The force expression looks as follows:
        
        .. math::
        
            \mathbf{F}(\mathbf{r}) = \mathbf{\nabla}S(\mathbf{r}) \cdot \begin{cases}
                0, & S(\mathbf{r}) < -h,\\
                \min(F_\text{max}, C (S(\mathbf{r}) + h)), & S(\mathbf{r}) \geqslant -h,\\
            \end{cases}

        where :math:`S` is the SDF of the wall, :math:`C`, :math:`F_\text{max}` and :math:`h` are parameters.
    )");

    
    py::handlers_class<XYZPlugin>(m, "XYZPlugin", pysim, R"(
        This plugin will dump positions of all the particles of the specified Particle Vector in the XYZ format.
   
        .. note::
            This plugin is inactive if postprocess is disabled
    )");
    
    py::handlers_class<XYZDumper>(m, "XYZDumper", pypost, R"(
        Postprocess side plugin of :any:`XYZPlugin`.
        Responsible for the I/O part.
    )");

    py::handlers_class<WallForceCollectorPlugin>(m, "WallForceCollector", pysim, R"(
        This plugin collects and average the total force exerted on a given wall.
        The result has 2 components:
        
            * bounce back: force necessary to the momentum change
            * frozen particles: total interaction force exerted on the frozen particles
    )");

    py::handlers_class<WallForceDumperPlugin>(m, "WallForceDumper", pypost, R"(
        Postprocess side plugin of :any:`WallForceCollector`.
        Responsible for the I/O part.
    )");

        
    
    
    
    m.def("__createAddForce", &PluginFactory::createAddForcePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "force"_a, R"(
        Create :any:`AddForce` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            force: extra force
    )");

    m.def("__createAddTorque", &PluginFactory::createAddTorquePlugin, 
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "torque"_a, R"(
        Create :any:`AddTorque` plugin
        
        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            torque: extra torque (per object)
    )");

    m.def("__createAnchorParticles", &PluginFactory::createAnchorParticlesPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "positions"_a, "velocities"_a, "pids"_a,
          "report_every"_a, "path"_a, R"(
        Create :any:`AnchorParticles` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            positions: positions (at given time) of the particles
            velocities: velocities (at given time) of the particles
            pids: global ids of the particles in the given particle vector
            report_every: report the time averaged force acting on the particles every this amount of timesteps
            path: folder where to dump the stats
    )");

    m.def("__createDensityControl", &PluginFactory::createDensityControlPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "file_name"_a, "pvs"_a, "target_density"_a,
          "region"_a, "resolution"_a, "level_lo"_a, "level_hi"_a, "level_space"_a,
          "Kp"_a, "Ki"_a, "Kd"_a, "tune_every"_a, "dump_every"_a, "sample_every"_a, R"(
        Create :any:`DensityControlPlugin`
        
        Args:
            name: name of the plugin
            file_name: output filename 
            pvs: list of :any:`ParticleVector` that we'll work with
            target_density: target number density (used only at boundaries of level sets)
            region: a scalar field which describes how to subdivide the domain. 
                    It must be continuous and differentiable, as the forces are in the gradient direction of this field
            resolution: grid resolution to represent the region field
            level_lo: lower level set to apply the controller on
            level_hi: highest level set to apply the controller on
            level_space: the size of one subregion in terms of level sets
            Kp, Ki, Kd: pid control parameters
            tune_every: update the forces every this amount of time steps
            dump_every: dump densities and forces in file ``filename``
            sample_every: sample to average densities every this amount of time steps
    )");

    m.def("__createDensityOutlet", &PluginFactory::createDensityOutletPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a, "number_density"_a,
          "region"_a, "resolution"_a, R"(
        Create :any:`DensityOutletPlugin`
        
        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            number_density: maximum number_density in the region
            region: a function that is negative in the concerned region and positive outside
            resolution: grid resolution to represent the region field
        
    )");

    m.def("__createPlaneOutlet", &PluginFactory::createPlaneOutletPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a, "plane"_a, R"(
        Create :any:`PlaneOutletPlugin`

        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            plane: Tuple (a, b, c, d). Particles are removed if `ax + by + cz + d >= 0`.
    )");

    m.def("__createRateOutlet", &PluginFactory::createRateOutletPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a, "mass_rate"_a,
          "region"_a, "resolution"_a, R"(
        Create :any:`RateOutletPlugin`
        
        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            mass_rate: total outlet mass rate in the region
            region: a function that is negative in the concerned region and positive outside
            resolution: grid resolution to represent the region field
        
    )");
    
    m.def("__createDumpAverage", &PluginFactory::createDumpAveragePlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a, "sample_every"_a, "dump_every"_a,
          "bin_size"_a = float3{1.0, 1.0, 1.0}, "channels"_a, "path"_a = "xdmf/", R"(
        Create :any:`Average3D` plugin
        
        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            sample_every: sample quantities every this many time-steps
            dump_every: write files every this many time-steps 
            bin_size: bin size for sampling. The resulting quantities will be *cell-centered*
            path: Path and filename prefix for the dumps. For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5
            channels: list of pairs name - type.
                Name is the channel (per particle) name. Always available channels are:
                    
                * 'velocity' with type "float4"
                
                Type is to provide the type of quantity to extract from the channel.                                            
                Type can also define a simple transformation from the channel internal structure                 
                to the datatype supported in HDF5 (i.e. scalar, vector, tensor)                                  
                Available types are:                                                                             
                                                                                                                
                * 'scalar': 1 float per particle
                * 'vector': 3 floats per particle
                * 'vector_from_float4': 4 floats per particle. 3 first floats will form the resulting vector
                * 'tensor6': 6 floats per particle, symmetric tensor in order xx, xy, xz, yy, yz, zz
                
    )");

    m.def("__createDumpAverageRelative", &PluginFactory::createDumpAverageRelativePlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a,
          "relative_to_ov"_a, "relative_to_id"_a,
          "sample_every"_a, "dump_every"_a,
          "bin_size"_a = float3{1.0, 1.0, 1.0}, "channels"_a, "path"_a = "xdmf/",
          R"(
              
        Create :any:`AverageRelative3D` plugin
                
        The arguments are the same as for createDumpAverage() with a few additions
        
        Args:
            relative_to_ov: take an object governing the frame of reference from this :any:`ObjectVector`
            relative_to_id: take an object governing the frame of reference with the specific ID
    )");

    m.def("__createDumpMesh", &PluginFactory::createDumpMeshPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "dump_every"_a, "path"_a, R"(
        Create :any:`MeshPlugin` plugin
        
        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<ov_name>_NNNNN.ply
    )");

    m.def("__createDumpObjectStats", &PluginFactory::createDumpObjStats, 
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "dump_every"_a, "path"_a, R"(
        Create :any:`ObjStats` plugin
        
        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<ov_name>_NNNNN.txt
    )");

    m.def("__createDumpParticles", &PluginFactory::createDumpParticlesPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "dump_every"_a,
          "channels"_a, "path"_a, R"(
        Create :any:`ParticleSenderPlugin` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            dump_every: write files every this many time-steps 
            path: Path and filename prefix for the dumps. For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5
            channels: list of pairs name - type.
                Name is the channel (per particle) name.
                The "velocity" and "id" channels are always activated.
                Type is to provide the type of quantity to extract from the channel.                                            
                Available types are:                                                                             
                                                                                                                
                * 'scalar': 1 float per particle
                * 'vector': 3 floats per particle
                * 'tensor6': 6 floats per particle, symmetric tensor in order xx, xy, xz, yy, yz, zz
                
    )");

    m.def("__createDumpParticlesWithRodData", &PluginFactory::createDumpParticlesWithRodDataPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "rv"_a, "dump_every"_a,
          "channels"_a, "path"_a, R"(
        Create :any:`ParticleSenderWithRodDataPlugin` plugin
        The interface is the same as :any:`createDumpParticles`

        Args:
            name: name of the plugin
            rv: :any:`RodVector` that we'll work with
            dump_every: write files every this many time-steps 
            path: Path and filename prefix for the dumps. For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5
            channels: list of pairs name - type.
                Name is the channel (per particle) name.
                The "velocity" and "id" channels are always activated.
                Type is to provide the type of quantity to extract from the channel.                                            
                Available types are:

                * 'scalar': 1 float per particle
                * 'vector': 3 floats per particle
                * 'tensor6': 6 floats per particle, symmetric tensor in order xx, xy, xz, yy, yz, zz
    )");

    m.def("__createDumpParticlesWithMesh", &PluginFactory::createDumpParticlesWithMeshPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "dump_every"_a,
          "channels"_a, "path"_a, R"(
        Create :any:`ParticleWithMeshSenderPlugin` plugin
        
        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            dump_every: write files every this many time-steps 
            path: Path and filename prefix for the dumps. For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5
            channels: list of pairs name - type.
                Name is the channel (per particle) name.
                The "velocity" and "id" channels are always activated.
                Type is to provide the type of quantity to extract from the channel.                                            
                Available types are:                                                                             
                                                                                                                
                * 'scalar': 1 float per particle
                * 'vector': 3 floats per particle
                * 'tensor6': 6 floats per particle, symmetric tensor in order xx, xy, xz, yy, yz, zz
                
    )");
    
    m.def("__createDumpXYZ", &PluginFactory::createDumpXYZPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "dump_every"_a, "path"_a, R"(
        Create :any:`XYZPlugin` plugin
        
        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<pv_name>_NNNNN.xyz
    )");

    m.def("__createExchangePVSFluxPlane", &PluginFactory::createExchangePVSFluxPlanePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv1"_a, "pv2"_a, "plane"_a, R"(
        Create :any:`ExchangePVSFluxPlane` plugin
        
        Args:
            name: name of the plugin
            pv1: :class:`ParticleVector` source
            pv2: :class:`ParticleVector` destination
            plane: 4 coefficients for the plane equation ax + by + cz + d >= 0
    )");


    m.def("__createForceSaver", &PluginFactory::createForceSaverPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, R"(
        Create :any:`ForceSaver` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
    )");

    m.def("__createImposeProfile", &PluginFactory::createImposeProfilePlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "low"_a, "high"_a, "velocity"_a, "kBT"_a, R"(
        Create :any:`ImposeProfile` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            low: the lower corner of the domain
            high: the higher corner of the domain
            velocity: target velocity
            kBT: temperature in the domain (appropriate Maxwell distribution will be used)
    )");

    m.def("__createImposeVelocity", &PluginFactory::createImposeVelocityPlugin,
        "compute_task"_a, "state"_a, "name"_a, "pvs"_a, "every"_a, "low"_a, "high"_a, "velocity"_a, R"(
        Create :any:`ImposeVelocity` plugin
        
        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            every: change the velocities once in **every** timestep
            low: the lower corner of the domain
            high: the higher corner of the domain
            velocity: target velocity
    )");

    m.def("__createMagneticOrientation", &PluginFactory::createMagneticOrientationPlugin,
          "compute_task"_a, "state"_a, "name"_a, "rov"_a, "moment"_a, "magneticFunction"_a, R"(
        Create :any:`MagneticOrientation` plugin
        
        Args:
            name: name of the plugin
            rov: :class:`RigidObjectVector` with which the magnetic field will interact
            moment: magnetic moment per object
            magneticFunction: a function that depends on time and returns a uniform (float3) magnetic field
    )");

    m.def("__createMembraneExtraForce", &PluginFactory::createMembraneExtraForcePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "forces"_a, R"(
        Create :any:`MembraneExtraForce` plugin
        
        Args:
            name: name of the plugin
            pv: :class:`ParticleVector` to which the force should be added
            forces: array of forces, one force (3 floats) per vertex in a single mesh
    )");

    m.def("__createObjectPortalDestination", &PluginFactory::createObjectPortalDestination,
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "src"_a, "dst"_a, "size"_a, "tag"_a, "interCommPtr"_a, R"(
        Create :any:`ObjectPortalDestination` plugin

        Args:
            name: name of the plugin
            ...
            ...
            ..
    )");

    m.def("__createObjectPortalSource", &PluginFactory::createObjectPortalSource,
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "src"_a, "dst"_a, "size"_a, "plane"_a, "tag"_a, "interCommPtr"_a, R"(
        Create :any:`ObjectPortalSource` plugin

        Args:
            name: name of the plugin
            ...
            ...
            ..
    )");

    m.def("__createObjectToParticlesPlugin", &PluginFactory::createObjectToParticlesPlugin,
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "pv"_a, "plane"_a, R"(
        Create :any:`ObjectPortalSource` plugin

        Args:
            name: name of the plugin
            ov: source object vector
            pv: target particle vector
            plane: plane `(a, b, c, d)` defined as `a*x + b*y + c*z + d == 0`
    )");


    m.def("__createParticleChannelSaver", &PluginFactory::createParticleChannelSaverPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "channelName"_a, "savedName"_a, R"(
        Create :any:`ParticleChannelSaver` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            channelName: the name of the source channel
            savedName: name of the extra channel
    )");

    m.def("__createParticleChecker", &PluginFactory::createParticleCheckerPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "check_every"_a, R"(
        Create :any:`ParticleChecker` plugin
        
        Args:
            name: name of the plugin
            check_every: check every this amount of time steps
    )");

    m.def("__createParticleDisplacement", &PluginFactory::createParticleDisplacementPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "update_every"_a, R"(
        Create :any:`ParticleDisplacementPlugin`
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            update_every: displacements are computed between positions separated by this amount of timesteps
    )");

    m.def("__createParticleDrag", &PluginFactory::createParticleDragPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "drag"_a, R"(
        Create :any:`ParticleDrag` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            drag: drag coefficient
    )");

    m.def("__createParticlePortalDestination", &PluginFactory::createParticlePortalDestination,
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "src"_a, "dst"_a, "size"_a, "tag"_a, "interCommPtr"_a, R"(
        Create :any:`ParticlePortalDestination` plugin

        Args:
            name: name of the plugin
            ...
            ...
            ..
    )");

    m.def("__createParticlePortalSource", &PluginFactory::createParticlePortalSource,
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "src"_a, "dst"_a, "size"_a, "tag"_a, "interCommPtr"_a, R"(
        Create :any:`ParticlePortalSource` plugin

        Args:
            name: name of the plugin
            ...
            ...
            ..
    )");

    m.def("__createPinObject", &PluginFactory::createPinObjPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "dump_every"_a, "path"_a, "velocity"_a, "angular_velocity"_a, R"(
        Create :any:`PinObject` plugin
        
        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<ov_name>_NNNNN.txt
            velocity: 3 floats, each component is the desired object velocity.
                If the corresponding component should not be restricted, set this value to :python:`PinObject::Unrestricted`
            angular_velocity: 3 floats, each component is the desired object angular velocity.
                If the corresponding component should not be restricted, set this value to :python:`PinObject::Unrestricted`
    )");

    m.def("__createPinRodExtremity", &PluginFactory::createPinRodExtremityPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "rv"_a, "segment_id"_a, "f_magn"_a, "target_direction"_a, R"(
        Create :any:`PinRodExtremity` plugin
        
        Args:
            name: name of the plugin
            rv: :any:`RodVector` that we'll work with
            segment_id: the segment to which the plugin is active
            f_magn: force magnitude
            target_direction: the direction in which the material frame tends to align
    )");

    m.def("__createRadialVelocityControl", &PluginFactory::createRadialVelocityControlPlugin,
          "compute_task"_a, "state"_a, "name"_a, "filename"_a, "pvs"_a, "minRadius"_a, "maxRadius"_a, 
          "sample_every"_a, "tune_every"_a, "dump_every"_a, "center"_a, "target_vel"_a, "Kp"_a, "Ki"_a, "Kd"_a, R"(
        Create :any:`VelocityControl` plugin
        
        Args:
            name: name of the plugin
            filename: dump file name 
            pvs: list of concerned :class:`ParticleVector`
            minRadius, maxRadius: only particles within this distance are considered 
            sample_every: sample velocity every this many time-steps
            tune_every: adapt the force every this many time-steps
            dump_every: write files every this many time-steps
            center: center of the radial coordinates
            target_vel: the target mean velocity of the particles at :math:`r=1`
            Kp, Ki, Kd: PID controller coefficients
    )");

    m.def("__createStats", &PluginFactory::createStatsPlugin,
          "compute_task"_a, "state"_a, "name"_a, "filename"_a="", "every"_a, R"(
        Create :any:`SimulationStats` plugin
        
        Args:
            name: name of the plugin
            filename: the stats will also be recorded to that file in a computer-friendly way
            every: report to standard output every that many time-steps
    )");

    m.def("__createTemperaturize", &PluginFactory::createTemperaturizePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "kBT"_a, "keepVelocity"_a, R"(
        Create :any:`Temperaturize` plugin

        Args:
            name: name of the plugin
            pv: the concerned :any:`ParticleVector`
            kBT: the target temperature
            keepVelocity: True for adding Maxwell distribution to the previous velocity; False to set the velocity to a Maxwell distribution.
    )");

    m.def("__createVelocityControl", &PluginFactory::createVelocityControlPlugin,
          "compute_task"_a, "state"_a, "name"_a, "filename"_a, "pvs"_a, "low"_a, "high"_a,
          "sample_every"_a, "tune_every"_a, "dump_every"_a, "target_vel"_a, "Kp"_a, "Ki"_a, "Kd"_a, R"(
        Create :any:`VelocityControl` plugin
        
        Args:
            name: name of the plugin
            filename: dump file name 
            pvs: list of concerned :class:`ParticleVector`
            low, high: boundaries of the domain of interest
            sample_every: sample velocity every this many time-steps
            tune_every: adapt the force every this many time-steps
            dump_every: write files every this many time-steps
            target_vel: the target mean velocity of the particles in the domain of interest
            Kp, Ki, Kd: PID controller coefficients
    )");

    m.def("__createVelocityInlet", &PluginFactory::createVelocityInletPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a,
          "implicit_surface_func"_a, "velocity_field"_a, "resolution"_a, "number_density"_a, "kBT"_a, R"(
        Create :any:`VelocityInlet` plugin
        
        Args:
            name: name of the plugin
            pv: the :any:`ParticleVector` that we ll work with 
            implicit_surface_func: a scalar field function that has the required surface as zero level set
            velocity_field: vector field that describes the velocity on the inlet (will be evaluated on the surface only)
            resolution: grid size used to discretize the surface
            number_density: number density of the inserted solvent
            kBT: temperature of the inserted solvent
    )");

    m.def("__createVirialPressurePlugin", &PluginFactory::createVirialPressurePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "regionFunc"_a, "h"_a, "dump_every"_a, "path"_a, R"(
        Create :any:`VirialPressure` plugin
        
        Args:
            name: name of the plugin
            pv: concerned :class:`ParticleVector`
            regionFunc: predicate for the concerned region; positive inside the region and negative outside
            h: grid size for representing the predicate onto a grid
            dump_every: report total pressure every this many time-steps
            path: the folder name in which the file will be dumped
    )");

    m.def("__createWallRepulsion", &PluginFactory::createWallRepulsionPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "wall"_a, "C"_a, "h"_a, "max_force"_a, R"(
        Create :any:`WallRepulsion` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            wall: :any:`Wall` that defines the repulsion
            C: :math:`C`  
            h: :math:`h`  
            max_force: :math:`F_{max}`  
    )");

    m.def("__createWallForceCollector", &PluginFactory::createWallForceCollectorPlugin, 
          "compute_task"_a, "state"_a, "name"_a, "wall"_a, "pvFrozen"_a, "sample_every"_a, "dump_every"_a, "filename"_a, R"(
        Create :any:`WallForceCollector` plugin
        
        Args:
            name: name of the plugin            
            wall: :any:`Wall` that we ll work with
            pvFrozen: corresponding frozen :any:`ParticleVector`
            sample_every: sample every this number of time steps
            dump_every: dump every this amount of timesteps
            filename: output filename
    )");
}

