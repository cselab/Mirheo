#include <extern/pybind11/include/pybind11/pybind11.h>
#include <extern/pybind11/include/pybind11/stl.h>

#include <plugins/factory.h>

#include "nodelete.h"

namespace py = pybind11;
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

    py::handlers_class<ExchangePVSFluxPlanePlugin>(m, "ExchangePVSFluxPlane", pysim, R"(
        This plugin exchanges particles from a particle vector crossing a given plane to another particle vector.
        A particle with position x, y, z has crossed the plane if ax + by + cz + d >= 0, where a, b, c and d are the coefficient 
        stored in the 'plane' variable
    )");

    py::handlers_class<ImposeProfilePlugin>(m, "ImposeProfile", pysim, R"(
        TODO
    )");
    
    py::handlers_class<ImposeVelocityPlugin>(m, "ImposeVelocity", pysim, R"(
        This plugin will add velocity to all the particles of the target PV in the specified area (rectangle) such that the average velocity equals to desired.
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

    py::handlers_class<ObjPositionsPlugin>(m, "ObjPositions", pysim, R"(
        This plugin will write the coordinates of the centers of mass of the objects of the specified Object Vector.
        If the objects are rigid bodies, also will be written: COM velocity, rotation, angular velocity, force, torque.
        
        The file format is the following:
        
        <object id> <simulation time> <COM>x3 [<quaternion>x4 <velocity>x3 <angular velocity>x3 <force>x3 <torque>x3]
        
        .. note::
            Note that all the written values are *instantaneous*
            
        .. note::
            This plugin is inactive if postprocess is disabled
    )");

    py::handlers_class<ParticleSenderPlugin>(m, "ParticleSenderPlugin", pysim, R"(
        This plugin will dump positions, velocities and optional attached data of all the particles of the specified Particle Vector.
        The data is dumped into hdf5 format. An additional xdfm file is dumped to describe the data and make it readable by visualization tools. 
    )");

    py::handlers_class<PinObjectPlugin>(m, "PinObject", pysim, R"(
        This plugin will fix center of mass positions (by axis) of all the objects of the specified Object Vector.
        If the objects are rigid bodies, rotatation may be restricted with this plugin as well.
        The *average* force or torque required to fix the positions or rotation are reported.
            
        .. note::
            This plugin is inactive if postprocess is disabled
    )");

    py::handlers_class<SimulationStats>(m, "SimulationStats", pysim, R"(
        This plugin will report aggregate quantities of all the particles in the simulation:
        total number of particles in the simulation, average temperature and momentum, maximum velocity magnutide of a particle
        and also the mean real time per step in milliseconds.
        
        .. note::
            This plugin is inactive if postprocess is disabled
    )");

    py::handlers_class<SimulationVelocityControl>(m, "VelocityControl", pysim, R"(
        This plugin applies a uniform force to all the particles of the target PVS in the specified area (rectangle).
        The force is adapted bvia a PID controller such that the velocity average of the particles matches the target average velocity.
    )");

    py::handlers_class<TemperaturizePlugin>(m, "Temperaturize", pysim, R"(
        TODO
    )");
    
    py::handlers_class<WallRepulsionPlugin>(m, "WallRepulsion", pysim, R"(
        This plugin will add force on all the particles that are nearby a specified wall. The motivation of this plugin is as follows.
        The particles of regular PVs are prevented from penetrating into the walls by Wall Bouncers.
        However, using Wall Bouncers with Object Vectors may be undesirable (e.g. in case of a very viscous membrane) or impossible (in case of rigid objects).
        In these cases one can use either strong repulsive potential between the object and the wall particle or alternatively this plugin.
        The advantage of the SDF-based repulsion is that small penetrations won't break the simulation.
        
        The force expression looks as follows:
        
        .. math::
        
            \mathbf{F} = \mathbf{\nabla}_{sdf} \cdot \begin{cases}
                0, & sdf < -h\\
                \min(F_{max}, C (sdf + h)), & sdf \geqslant -h\\
            \end{cases}
    )");
    
    py::handlers_class<XYZPlugin>(m, "XYZPlugin", pysim, R"(
        This plugin will dump positions of all the particles of the specified Particle Vector in the XYZ format.
   
        .. note::
            This plugin is inactive if postprocess is disabled
    )");
    
    
    py::handlers_class<PostprocessStats>(m, "PostprocessStats", pypost);
    py::handlers_class<UniformCartesianDumper>(m, "UniformCartesianDumper", pypost);
    py::handlers_class<XYZDumper>(m, "XYZDumper", pypost);
    py::handlers_class<ParticleDumperPlugin>(m, "ParticleDumperPlugin", pypost);
    py::handlers_class<MeshDumper>(m, "MeshDumper", pypost);
    py::handlers_class<ObjPositionsDumper>(m, "ObjPositionsDumper", pypost);
    py::handlers_class<ReportPinObjectPlugin>(m, "ReportPinObject", pypost);
    py::handlers_class<PostprocessVelocityControl>(m, "PostprocessVelocityControl", pypost);
    
    m.def("__createImposeVelocity", &PluginFactory::createImposeVelocityPlugin,
        "compute_task"_a, "name"_a, "pv"_a, "every"_a, "low"_a, "high"_a, "velocity"_a, R"(
        Create :any:`ImposeVelocity` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            every: change the velocities once in **every** timestep
            low: the lower corner of the domain
            high: the higher corner of the domain
            velocity: target velocity
    )");
    m.def("__createTemperaturize", &PluginFactory::createTemperaturizePlugin, R"(
        TODO
    )");
    m.def("__createAddForce", &PluginFactory::createAddForcePlugin,
         "compute_task"_a, "name"_a, "pv"_a, "force"_a, R"(
        Create :any:`AddForce` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            force: extra force
    )");
    m.def("__createAddTorque", &PluginFactory::createAddTorquePlugin, 
          "compute_task"_a, "name"_a, "ov"_a, "torque"_a, R"(
        Create :any:`AddTorque` plugin
        
        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            torque: extra torque (per object)
    )");
    m.def("__createImposeProfile", &PluginFactory::createImposeProfilePlugin, 
          "compute_task"_a, "name"_a, "pv"_a, "low"_a, "high"_a, "velocity"_a, "kbt"_a, R"(
        Create :any:`ImposeProfile` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            low: the lower corner of the domain
            high: the higher corner of the domain
            velocity: target velocity
            kbt: temperature in the domain (appropriate Maxwell distribution will be used)
    )");
    m.def("__createWallRepulsion", &PluginFactory::createWallRepulsionPlugin, 
          "compute_task"_a, "name"_a, "pv"_a, "wall"_a, "C"_a, "h"_a, "max_force"_a, R"(
        Create :any:`WallRepulsion` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            wall: :any:`Wall` that defines the repulsion
            C: :math:`C`  
            h: :math:`h`  
            max_force: :math:`F_{max}`  
    )");
    m.def("__createStats", &PluginFactory::createStatsPlugin,
          "compute_task"_a, "name"_a, "filename"_a, "every"_a, R"(
        Create :any:`SimulationStats` plugin
        
        Args:
            name: name of the plugin
            filename: the stats will also be recorded to that file in a computer-friendly way
            every: report to standard output every that many time-steps
    )");
    m.def("__createDumpAverage", &PluginFactory::createDumpAveragePlugin, 
          "compute_task"_a, "name"_a, "pvs"_a, "sample_every"_a, "dump_every"_a,
          "bin_size"_a = PyTypes::float3{1.0, 1.0, 1.0}, "channels"_a, "path"_a = "xdmf/", R"(
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
                    
                * 'velocity' with type "float8"             
                * 'force' with type "float4"
                
                Type is to provide the type of quantity to extract from the channel.                                            
                Type can also define a simple transformation from the channel internal structure                 
                to the datatype supported in HDF5 (i.e. scalar, vector, tensor)                                  
                Available types are:                                                                             
                                                                                                                
                * 'scalar': 1 float per particle                                                                   
                * 'vector': 3 floats per particle                                                                  
                * 'vector_from_float4': 4 floats per particle. 3 first floats will form the resulting vector       
                * 'vector_from_float8' 8 floats per particle. 5th, 6th, 7th floats will form the resulting vector. 
                    This type is primarity made to be used with velocity since it is stored together with          
                    the coordinates as 8 consecutive float numbers: (x,y,z) coordinate, followed by 1 padding value
                    and then (x,y,z) velocity, followed by 1 more padding value                                    
                * 'tensor6': 6 floats per particle, symmetric tensor in order xx, xy, xz, yy, yz, zz
                
    )");
    m.def("__createDumpAverageRelative", &PluginFactory::createDumpAverageRelativePlugin, 
          "compute_task"_a, "name"_a, "pvs"_a,
          "relative_to_ov"_a, "relative_to_id"_a,
          "sample_every"_a, "dump_every"_a,
          "bin_size"_a = PyTypes::float3{1.0, 1.0, 1.0}, "channels"_a, "path"_a = "xdmf/",
          R"(
              
        Create :any:`AverageRelative3D` plugin
                
        The arguments are the same as for createDumpAverage() with a few additions
        
        Args:
            relative_to_ov: take an object governing the frame of reference from this :any:`ObjectVector`
            relative_to_id: take an object governing the frame of reference with the specific ID
    )");

    m.def("__createDumpParticles", &PluginFactory::createDumpParticlesPlugin, 
          "compute_task"_a, "name"_a, "pv"_a, "dump_every"_a,
          "channels"_a, "path"_a, R"(
        Create :any:`ParticleSenderPlugin` plugin
        
        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            dump_every: write files every this many time-steps 
            path: Path and filename prefix for the dumps. For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5
            channels: list of pairs name - type.
                Name is the channel (per particle) name.
                The "velocity" channel is always activated by default.
                Type is to provide the type of quantity to extract from the channel.                                            
                Available types are:                                                                             
                                                                                                                
                * 'scalar': 1 float per particle
                * 'vector': 3 floats per particle
                * 'tensor6': 6 floats per particle, symmetric tensor in order xx, xy, xz, yy, yz, zz
                
    )");
    
    m.def("__createDumpXYZ", &PluginFactory::createDumpXYZPlugin, 
          "compute_task"_a, "name"_a, "pv"_a, "dump_every"_a, "path"_a, R"(
        Create :any:`XYZPlugin` plugin
        
        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<pv_name>_NNNNN.xyz
    )");
    m.def("__createDumpMesh", &PluginFactory::createDumpMeshPlugin, 
          "compute_task"_a, "name"_a, "ov"_a, "dump_every"_a, "path"_a, R"(
        Create :any:`MeshPlugin` plugin
        
        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<ov_name>_NNNNN.ply
    )");
    m.def("__createDumpObjectStats", &PluginFactory::createDumpObjPosition, 
          "compute_task"_a, "name"_a, "ov"_a, "dump_every"_a, "path"_a, R"(
        Create :any:`ObjPositions` plugin
        
        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<ov_name>_NNNNN.txt
    )");
    m.def("__createPinObject", &PluginFactory::createPinObjPlugin, 
          "compute_task"_a, "name"_a, "ov"_a, "dump_every"_a, "path"_a, "pin_translation"_a, "pin_rotation"_a, R"(
        Create :any:`PinObject` plugin
        
        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<ov_name>_NNNNN.txt
            pin_translation: 3 integers; 0 means that motion along the corresponding axis is unrestricted,
                1 means fixed position wrt to the axis
            pin_rotation: 3 integers; 0 means that rotation along the corresponding axis is unrestricted,
                1 means fixed rotation wrt to the axis
    )");
    m.def("__createVelocityControl", &PluginFactory::createSimulationVelocityControlPlugin,
          "compute_task"_a, "name"_a, "filename"_a, "pvs"_a, "low"_a, "high"_a, "sampleEvery"_a, "dumpEvery"_a, "targetVel"_a, "Kp"_a, "Ki"_a, "Kd"_a, R"(
        Create :any:`VelocityControl` plugin
        
        Args:
            name: name of the plugin
            filename: dump file name 
            pvs: list of concerned :class:`ParticleVector`
            low, high: boundaries of the domain of interest
            sampleEvery: sample and adapt force every this many time-steps
            dumpEvery: write files every this many time-steps
            targetVel: the target mean velocity of the particles in the domain of interest
            Kp, Ki, Kd: PID controller coefficients
    )");
    m.def("__createExchangePVSFluxPlane", &PluginFactory::createExchangePVSFluxPlanePlugin,
          "compute_task"_a, "name"_a, "pv1"_a, "pv2"_a, "plane"_a, R"(
        Create :any:`ExchangePVSFluxPlane` plugin
        
        Args:
            name: name of the plugin
            pv1: :class:`ParticleVector` source
            pv2: :class:`ParticleVector` destination
            plane: 4 coefficients for the plane equation ax + by + cz + d >= 0
    )");
    m.def("__createMembraneExtraForce", &PluginFactory::createMembraneExtraForcePlugin,
          "compute_task"_a, "name"_a, "pv"_a, "forces"_a, R"(
        Create :any:`MembraneExtraForce` plugin
        
        Args:
            name: name of the plugin
            pv: :class:`ParticleVector` to which the force should be added
            forces: array of forces, one force (3 floats) per vertex in a single mesh
    )");
}

