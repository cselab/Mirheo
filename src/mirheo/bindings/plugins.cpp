// Copyright 2020 ETH Zurich. All Rights Reserved.

#include "plugins.h"
#include "class_wrapper.h"

#include <mirheo/plugins/factory.h>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace mirheo
{

using namespace pybind11::literals;

void exportPlugins(py::module& m)
{
    py::handlers_class<SimulationPlugin>  pysim(m, "SimulationPlugin", R"(
        Base simulation plugin class
    )");

    py::handlers_class<PostprocessPlugin> pypost(m, "PostprocessPlugin", R"(
        Base postprocess plugin class
    )");


    m.def("__createAddFourRollMillForce", &plugin_factory::createAddFourRollMillForcePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "intensity"_a, R"(
        This plugin will add a force :math:`\mathbf{f} = (A \sin x \cos y, A \cos x \sin y, 0)` to each particle of a specific PV every time-step.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            intensity: The intensity of the force
    )");

    m.def("__createAddForce", &plugin_factory::createAddForcePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "force"_a, R"(
        This plugin will add constant force :math:`\mathbf{F}_{extra}` to each particle of a specific PV every time-step.
        Is is advised to only use it with rigid objects, since Velocity-Verlet integrator with constant pressure can do the same without any performance penalty.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            force: extra force
    )");

    m.def("__createAddForceField",
          py::overload_cast<bool,const MirState*,std::string,ParticleVector*,std::function<real3(real3)>,real3>
          (&plugin_factory::createAddForceFieldPlugin),
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "force_field"_a, "h"_a, R"(
        Add a force to each particle of a specific PV every time-step.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            force_field: force field
            h: grid spacing used to discretize the force field
    )");

    m.def("__createAddForceField",
          py::overload_cast<bool,const MirState*,std::string,ParticleVector*,std::string,real3>
          (&plugin_factory::createAddForceFieldPlugin),
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "force_field_filename"_a, "h"_a, R"(
        Add a force to each particle of a specific PV every time-step.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            force_field_filename: file that contains the force field on a cartesian grid. Same format as Sdf for walls but with 4 components per grid point (only the first three are used).
            h: grid spacing used to discretize the force field
    )");

    m.def("__createAddPotentialForce",
          py::overload_cast<bool,const MirState*,std::string,ParticleVector*,std::function<real(real3)>,real3>
          (&plugin_factory::createAddPotentialForcePlugin),
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "potential_field"_a, "h"_a, R"(
        Add a force :math:`\mathbf{F}_{extra}` to each particle of a specific PV every time-step.
        The force is the negative gradient of a potential field at the particle position.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            potential_field: potential field
            h: grid spacing used to discretize the potential field
    )");

    m.def("__createAddPotentialForce",
          py::overload_cast<bool,const MirState*,std::string,ParticleVector*,std::string,real3>
          (&plugin_factory::createAddPotentialForcePlugin),
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "potential_field_filename"_a, "h"_a, R"(
        Add a force :math:`\mathbf{F}_{extra}` to each particle of a specific PV every time-step.
        The force is the negative gradient of a potential field at the particle position.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            potential_field_filename: file that contains the potential field on a cartesian grid. Same format as Sdf for walls.
            h: grid spacing used to discretize the potential field
    )");

    m.def("__createAddReversePoiseuilleForce", &plugin_factory::createAddReversePoiseuilleForcePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "force"_a, "flip_direction"_a, R"(
        This plugin will add constant force :math:`\mathbf{F}_{extra}` to each particle of a specific PV every time-step.
        The force is flipped if the position is in the upper half along the flip_direction.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            force: extra force
            flip_direction: either x, y or z. The direction along with the sign of the force changes.
    )");

    m.def("__createAddSinusoidalForce", &plugin_factory::createAddSinusoidalForcePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "magnitude"_a, "wave_number"_a, R"(
        This plugin will add sinusoidal force :math:`\mathbf{F}(\mathbf{r}) = A \sin\left( 2\pi k r_y / L_y\right)` to each particle of a specific PV every time-step, where :math:`L_y` is the dimension of the domain along :math`y`.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            magnitude: coefficient :math:`A`
            wave_number: mode :math:`k` (integer)
    )");

    m.def("__createAddTorque", &plugin_factory::createAddTorquePlugin,
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "torque"_a, R"(
        This plugin will add constant torque :math:`\mathbf{T}_{extra}` to each *object* of a specific OV every time-step.

        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            torque: extra torque (per object)
    )");

    m.def("__createAnchorParticles", &plugin_factory::createAnchorParticlesPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "positions"_a, "velocities"_a, "pids"_a,
          "report_every"_a, "path"_a, R"(
        This plugin will set a given particle at a given position and velocity.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            positions: positions (at given time) of the particles
            velocities: velocities (at given time) of the particles
            pids: global ids of the particles in the given particle vector
            report_every: report the time averaged force acting on the particles every this amount of timesteps
            path: folder where to dump the stats
    )");

    m.def("__createBerendsenThermostat", &plugin_factory::createBerendsenThermostatPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a,
          "tau"_a, "kBT"_a, "increaseIfLower"_a=true, R"(
        Berendsen thermostat.

        On each time step the velocities of all particles in given particle vectors are multiplied by the following factor:

        .. math::

            \lambda = \sqrt{1 + \frac{\Delta t}{\tau} \left( \frac{T_0}{T} - 1 \right)}

        where :math:`\Delta t` is a time step, :math:`\tau` relaxation time,
        :math:`T` current temperature, :math:`T_0` target temperature.

        Reference: `Berendsen et al. (1984) <https://aip.scitation.org/doi/10.1063/1.448118>`_

        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` objects to apply the thermostat to
            tau: relaxation time :math:`\tau`
            kBT: target thermal energy :math:`k_B T_0`
            increaseIfLower: whether to increase the temperature if it's lower than the target temperature
    )");

    m.def("__createDensityControl", &plugin_factory::createDensityControlPlugin,
          "compute_task"_a, "state"_a, "name"_a, "file_name"_a, "pvs"_a, "target_density"_a,
          "region"_a, "resolution"_a, "level_lo"_a, "level_hi"_a, "level_space"_a,
          "Kp"_a, "Ki"_a, "Kd"_a, "tune_every"_a, "dump_every"_a, "sample_every"_a, R"(
        This plugin applies forces to a set of particle vectors in order to get a constant density.

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

    m.def("__createDensityOutlet", &plugin_factory::createDensityOutletPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a, "number_density"_a,
          "region"_a, "resolution"_a, R"(
        This plugin removes particles from a set of :any:`ParticleVector` in a given region if the number density is larger than a given target.

        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            number_density: maximum number_density in the region
            region: a function that is negative in the concerned region and positive outside
            resolution: grid resolution to represent the region field

    )");

    m.def("__createPlaneOutlet", &plugin_factory::createPlaneOutletPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a, "plane"_a, R"(
        This plugin removes all particles from a set of :any:`ParticleVector` that are on the non-negative side of a given plane.

        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            plane: Tuple (a, b, c, d). Particles are removed if `ax + by + cz + d >= 0`.
    )");

    m.def("__createRateOutlet", &plugin_factory::createRateOutletPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a, "mass_rate"_a,
          "region"_a, "resolution"_a, R"(
        This plugin removes particles from a set of :any:`ParticleVector` in a given region at a given mass rate.

        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            mass_rate: total outlet mass rate in the region
            region: a function that is negative in the concerned region and positive outside
            resolution: grid resolution to represent the region field

    )");

    m.def("__createDumpAverage", &plugin_factory::createDumpAveragePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a, "sample_every"_a, "dump_every"_a,
          "bin_size"_a = real3{1.0, 1.0, 1.0}, "channels"_a, "path"_a = "xdmf/", R"(
        This plugin will project certain quantities of the particle vectors on the grid (by simple binning),
        perform time-averaging of the grid and dump it in `XDMF <http://www.xdmf.org/index.php/XDMF_Model_and_Format>`_ format
        with `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ backend.
        The quantities of interest are represented as *channels* associated with particles vectors.
        Some interactions, integrators, etc. and more notable plug-ins can add to the Particle Vectors per-particles arrays to hold different values.
        These arrays are called *channels*.
        Any such channel may be used in this plug-in, however, user must explicitely specify the type of values that the channel holds.
        Particle number density is used to correctly average the values, so it will be sampled and written in any case into the field "number_densities".

        .. note::
            This plugin is inactive if postprocess is disabled

        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            sample_every: sample quantities every this many time-steps
            dump_every: write files every this many time-steps
            bin_size: bin size for sampling. The resulting quantities will be *cell-centered*
            path: Path and filename prefix for the dumps. For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5
            channels: list of channel names. See :ref:`user-pv-reserved`.
    )");

    m.def("__createDumpAverageRelative", &plugin_factory::createDumpAverageRelativePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pvs"_a,
          "relative_to_ov"_a, "relative_to_id"_a,
          "sample_every"_a, "dump_every"_a,
          "bin_size"_a = real3{1.0, 1.0, 1.0}, "channels"_a, "path"_a = "xdmf/",
          R"(
        This plugin acts just like the regular flow dumper, with one difference.
        It will assume a coordinate system attached to the center of mass of a specific object.
        In other words, velocities and coordinates sampled correspond to the object reference frame.

        .. note::
            Note that this plugin needs to allocate memory for the grid in the full domain, not only in the corresponding MPI subdomain.
            Therefore large domains will lead to running out of memory

        .. note::
            This plugin is inactive if postprocess is disabled

        The arguments are the same as for createDumpAverage() with a few additions:

        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            sample_every: sample quantities every this many time-steps
            dump_every: write files every this many time-steps
            bin_size: bin size for sampling. The resulting quantities will be *cell-centered*
            path: Path and filename prefix for the dumps. For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5
            channels: list of channel names. See :ref:`user-pv-reserved`.
            relative_to_ov: take an object governing the frame of reference from this :any:`ObjectVector`
            relative_to_id: take an object governing the frame of reference with the specific ID
    )");

    m.def("__createDumpMesh", &plugin_factory::createDumpMeshPlugin,
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "dump_every"_a, "path"_a, R"(
        This plugin will write the meshes of all the object of the specified Object Vector in a `PLY format <https://en.wikipedia.org/wiki/PLY_(file_format)>`_.

        .. note::
            This plugin is inactive if postprocess is disabled

        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<ov_name>_NNNNN.ply
    )");

    m.def("__createDumpObjectStats", &plugin_factory::createDumpObjStats,
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "dump_every"_a, "filename"_a, R"(
        This plugin will write the coordinates of the centers of mass of the objects of the specified Object Vector.
        Instantaneous quantities (COM velocity, angular velocity, force, torque) are also written.
        If the objects are rigid bodies, also will be written the quaternion describing the rotation.
        The `type id` field is also dumped if the objects have this field activated (see :class:`~libmirheo.InitialConditions.MembraneWithTypeId`).

        The file format is the following:

        <object id> <simulation time> <COM>x3 [<quaternion>x4] <velocity>x3 <angular velocity>x3 <force>x3 <torque>x3 [<type id>]

        .. note::
            Note that all the written values are *instantaneous*

        .. note::
            This plugin is inactive if postprocess is disabled

        Args:
            name: Name of the plugin.
            ov: :any:`ObjectVector` that we'll work with.
            dump_every: Write files every this many time-steps.
            filename: The name of the resulting csv file.
    )");

    m.def("__createDumpParticles", &plugin_factory::createDumpParticlesPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "dump_every"_a,
          "channel_names"_a, "path"_a, R"(
        This plugin will dump positions, velocities and optional attached data of all the particles of the specified Particle Vector.
        The data is dumped into hdf5 format. An additional xdfm file is dumped to describe the data and make it readable by visualization tools.
        If a channel from object data or bisegment data is provided, the data will be scattered to particles before being dumped as normal particle data.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            dump_every: write files every this many time-steps
            channel_names: list of channel names to be dumped.
            path: Path and filename prefix for the dumps. For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5
    )");

    m.def("__createDumpParticlesWithMesh", &plugin_factory::createDumpParticlesWithMeshPlugin,
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "dump_every"_a,
          "channel_names"_a, "path"_a, R"(
        This plugin will dump positions, velocities and optional attached data of all the particles of the specified Object Vector, as well as connectivity information.
        The data is dumped into hdf5 format. An additional xdfm file is dumped to describe the data and make it readable by visualization tools.

        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            dump_every: write files every this many time-steps
            channel_names: list of channel names to be dumped.
            path: Path and filename prefix for the dumps. For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5
    )");

    m.def("__createDumpParticlesWithPolylines", &plugin_factory::createDumpParticlesWithPolylinesPlugin,
          "compute_task"_a, "state"_a, "name"_a, "cv"_a, "dump_every"_a,
          "channel_names"_a, "path"_a, R"(
        This plugin will dump positions, velocities and optional attached data of all the particles of the specified
        ChainVector, as well as connectivity information representing polylines.
        The data is dumped into hdf5 format.
        An additional xdfm file is dumped to describe the data and make it readable by visualization tools.

        Args:
            name: name of the plugin.
            cv: :any:`ChainVector` to be dumped.
            dump_every: write files every this many time-steps.
            channel_names: list of channel names to be dumped.
            path: Path and filename prefix for the dumps. For every dump two files will be created: <path>_NNNNN.xmf and <path>_NNNNN.h5
    )");

    m.def("__createDumpXYZ", &plugin_factory::createDumpXYZPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "dump_every"_a, "path"_a, R"(
        This plugin will dump positions of all the particles of the specified Particle Vector in the XYZ format.

        .. note::
            This plugin is inactive if postprocess is disabled

        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<pv_name>_NNNNN.xyz
    )");

    m.def("__createExchangePVSFluxPlane", &plugin_factory::createExchangePVSFluxPlanePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv1"_a, "pv2"_a, "plane"_a, R"(
        This plugin exchanges particles from a particle vector crossing a given plane to another particle vector.
        A particle with position x, y, z has crossed the plane if ax + by + cz + d >= 0, where a, b, c and d are the coefficient
        stored in the 'plane' variable

        Args:
            name: name of the plugin
            pv1: :class:`ParticleVector` source
            pv2: :class:`ParticleVector` destination
            plane: 4 coefficients for the plane equation ax + by + cz + d >= 0
    )");

    m.def("__createExpMovingAverage", &plugin_factory::createExpMovingAveragePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "alpha"_a, "src_channel_name"_a, "ema_channel_name"_a, R"(
        Compute the exponential moving average (EMA) of the given channel of a :class:`ParticleVector` and stores it in the new channel "ema_channel_name".

        Args:
            name: name of the plugin
            pv: :class:`ParticleVector` source
            alpha: EMA coefficient. must be in [0, 1].
            src_channel_name: The name of the source channel.
            ema_channel_name: The name of the new EMA channel.
    )");

    m.def("__createExternalMagneticTorque", &plugin_factory::createExternalMagneticTorquePlugin,
          "compute_task"_a, "state"_a, "name"_a, "rov"_a, "moment"_a, "magneticFunction"_a, R"(
        This plugin gives a magnetic moment :math:`\mathbf{M}` to every rigid objects in a given :any:`RigidObjectVector`.
        It also models a uniform magnetic field :math:`\mathbf{B}` (varying in time) and adds the induced torque to the objects according to:

        .. math::

            \mathbf{T} = \mathbf{M} \times \mathbf{B}

        The magnetic field is passed as a function from python.
        The function must take a real (time) as input and output a tuple of three reals (magnetic field).

        Args:
            name: name of the plugin
            rov: :class:`RigidObjectVector` with which the magnetic field will interact
            moment: magnetic moment per object
            magneticFunction: a function that depends on time and returns a uniform (real3) magnetic field
    )");

    m.def("__createForceSaver", &plugin_factory::createForceSaverPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, R"(
        This plugin creates an extra channel per particle inside the given particle vector named 'forces'.
        It copies the total forces at each time step and make it accessible by other plugins.
        The forces are stored in an array of real3.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
    )");

    m.def("__createImposeProfile", &plugin_factory::createImposeProfilePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "low"_a, "high"_a, "velocity"_a, "kBT"_a, R"(
        This plugin will set the velocity of each particle inside a given domain to a target velocity with an additive term
        drawn from Maxwell distribution of the given temperature.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            low: the lower corner of the domain
            high: the higher corner of the domain
            velocity: target velocity
            kBT: temperature in the domain (appropriate Maxwell distribution will be used)
    )");

    m.def("__createImposeVelocity", &plugin_factory::createImposeVelocityPlugin,
        "compute_task"_a, "state"_a, "name"_a, "pvs"_a, "every"_a, "low"_a, "high"_a, "velocity"_a, R"(
        This plugin will add velocity to all the particles of the target PV in the specified area (rectangle) such that the average velocity equals to desired.

        Args:
            name: name of the plugin
            pvs: list of :any:`ParticleVector` that we'll work with
            every: change the velocities once in **every** timestep
            low: the lower corner of the domain
            high: the higher corner of the domain
            velocity: target velocity
    )");

    m.def("__createMagneticDipoleInteractions", &plugin_factory::createMagneticDipoleInteractionsPlugin,
          "compute_task"_a, "state"_a, "name"_a, "rov"_a, "moment"_a, "mu0"_a, "periodic"_a=true, R"(
        This plugin computes the forces and torques resulting from pairwise dipole-dipole interactions between rigid objects.
        All rigid objects are assumed to be the same with a constant magnetic moment in their frame of reference.

        Args:
            name: name of the plugin
            rov: :class:`RigidObjectVector` with which the magnetic field will interact
            moment: magnetic moment per object
            mu0: magnetic permeability of the medium
            periodic: if True, compute the interactions from the closest periodic image of each object.
    )");

    m.def("__createMembraneExtraForce", &plugin_factory::createMembraneExtraForcePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "forces"_a, R"(
        This plugin adds a given external force to a given membrane.
        The force is defined vertex wise and does not depend on position.
        It is the same for all membranes belonging to the same particle vector.

        Args:
            name: name of the plugin
            pv: :class:`ParticleVector` to which the force should be added
            forces: array of forces, one force (3 reals) per vertex in a single mesh
    )");

    m.def("__createMsd", &plugin_factory::createMsdPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "start_time"_a, "end_time"_a, "dump_every"_a, "path"_a, R"(
        This plugin computes the mean square displacement of th particles of a given :any:`ParticleVector`.
        The reference position` is that of the given :any:`ParticleVector` at the given start time.

        Args:
            name: Name of the plugin.
            pv: Concerned :class:`ParticleVector`.
            start_time: Simulation time of the reference positions.
            end_time: End time until which to compute the MSD.
            dump_every: Report MSD every this many time-steps.
            path: The folder name in which the file will be dumped.
    )");

    m.def("__createParticleChannelAverager", &plugin_factory::createParticleChannelAveragerPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "channelName"_a, "averageName"_a, "updateEvery"_a, R"(
        This plugin averages a channel (per particle data) inside the given particle vector and saves it to a new channel.
        This new channel (containing the averaged data) is updated every fixed number of time steps.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            channelName: The name of the source channel.
            averageName: The name of the average channel.
            updateEvery: reinitialize the averages every this number of steps.
    )");

    m.def("__createParticleChannelSaver", &plugin_factory::createParticleChannelSaverPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "channelName"_a, "savedName"_a, R"(
        This plugin creates an extra channel per particle inside the given particle vector with a given name.
        It copies the content of an extra channel of pv at each time step and make it accessible by other plugins.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            channelName: the name of the source channel
            savedName: name of the extra channel
    )");

    m.def("__createParticleChecker", &plugin_factory::createParticleCheckerPlugin,
          "compute_task"_a, "state"_a, "name"_a, "check_every"_a, R"(
        This plugin will check the positions and velocities of all particles in the simulation every given time steps.
        To be used for debugging purpose.

        Args:
            name: name of the plugin
            check_every: check every this amount of time steps
    )");

    m.def("__createParticleDisplacement", &plugin_factory::createParticleDisplacementPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "update_every"_a, R"(
        This plugin computes and save the displacement of the particles within a given particle vector.
        The result is stored inside the extra channel "displacements" as an array of real3.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            update_every: displacements are computed between positions separated by this amount of timesteps
    )");

    m.def("__createParticleDrag", &plugin_factory::createParticleDragPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "drag"_a, R"(
        This plugin will add drag force :math:`\mathbf{f} = - C_d \mathbf{u}` to each particle of a specific PV every time-step.

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            drag: drag coefficient
    )");

    py::handlers_class<plugin_factory::PinObjectMock>(m, "PinObject", pysim, R"(
        Contains the special value `Unrestricted` for unrestricted axes in :any:`createPinObject`.
    )")
        .def_property_readonly_static("Unrestricted", [](py::object) { return plugin_factory::PinObjectMock::Unrestricted; }, R"(
        Unrestricted
    )");

    m.def("__createPinObject", &plugin_factory::createPinObjPlugin,
          "compute_task"_a, "state"_a, "name"_a, "ov"_a, "dump_every"_a, "path"_a, "velocity"_a, "angular_velocity"_a, R"(
        This plugin will impose given velocity as the center of mass velocity (by axis) of all the objects of the specified Object Vector.
        If the objects are rigid bodies, rotation may be restricted with this plugin as well.
        The *time-averaged* force and/or torque required to impose the velocities and rotations are reported in the dumped file, with the following format:

        <object id> <simulation time> <force>x3 [<torque>x3]

        .. note::
            This plugin is inactive if postprocess is disabled

        Args:
            name: name of the plugin
            ov: :any:`ObjectVector` that we'll work with
            dump_every: write files every this many time-steps
            path: the files will look like this: <path>/<ov_name>.csv
            velocity: 3 reals, each component is the desired object velocity.
                If the corresponding component should not be restricted, set this value to :python:`PinObject::Unrestricted`
            angular_velocity: 3 reals, each component is the desired object angular velocity.
                If the corresponding component should not be restricted, set this value to :python:`PinObject::Unrestricted`
    )");

    m.def("__createPinRodExtremity", &plugin_factory::createPinRodExtremityPlugin,
          "compute_task"_a, "state"_a, "name"_a, "rv"_a, "segment_id"_a, "f_magn"_a, "target_direction"_a, R"(
        This plugin adds a force on a given segment of all the rods in a :any:`RodVector`.
        The force has the form deriving from the potential

        .. math::

            E = k \left( 1 - \cos \theta \right),

        where :math:`\theta` is the angle between the material frame and a given direction (projected on the concerned segment).
        Note that the force is applied only on the material frame and not on the center line.

        Args:
            name: name of the plugin
            rv: :any:`RodVector` that we'll work with
            segment_id: the segment to which the plugin is active
            f_magn: force magnitude
            target_direction: the direction in which the material frame tends to align
    )");

    m.def("__createRdf", &plugin_factory::createRdfPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "max_dist"_a, "nbins"_a, "basename"_a, "every"_a, R"(
        Compute the radial distribution function (RDF) of a given :any:`ParticleVector`.
        For simplicity, particles that are less that `max_dist` from the subdomain border are not counted.

        Args:
            name: Name of the plugin.
            pv: The :any:`ParticleVector` that we ant to compute the RDF from.
            max_dist: The RDF will be computed on the interval [0, max_dist]. Must be strictly less than half the minimum size of one subdomain.
            nbins: The RDF is computed on nbins bins.
            basename: Each RDF dump will be dumped in csv format to <basename>-XXXXX.csv.
            every: Computes and dump the RDF every this amount of timesteps.
    )");

    m.def("__createRmacf", &plugin_factory::createRmacfPlugin,
          "compute_task"_a, "state"_a, "name"_a, "cv"_a, "start_time"_a, "end_time"_a, "dump_every"_a, "path"_a, R"(
        This plugin computes the mean Rouse mode autocorrelation over time from a given :any:`ChainVector`.
        The reference modes are that of the :any:`ChainVector` at the given start time.

        Args:
            name: Name of the plugin.
            cv: Concerned :class:`ChainVector`.
            start_time: Simulation time of the reference velocities.
            end_time: End time until which to compute the RMACF.
            dump_every: Report the RMACF every this many time-steps.
            path: The folder name in which the file will be dumped.
    )");

    m.def("__createShearField", &plugin_factory::createShearFieldPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "shear"_a, "origin"_a, "sf_channel_name"_a, R"(
        This plugin computes the shear velocity field at every particle's position and stores it in a channel vector.

        Args:
            name: Name of the plugin.
            pv: Concerned :class:`ParticleVector`.
            shear: Shear tensor.
            origin: Point in space with zero velocity.
            sf_channel_name: Name of the channel that will contain the shear field.
    )");

    m.def("__createSinusoidalField", &plugin_factory::createSinusoidalFieldPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "magnitude"_a, "wave_number"_a, "sf_channel_name"_a, R"(
        This plugin computes a sinusoidal velocity field at every particle's position and stores it in a channel vector.

        Args:
            name: Name of the plugin.
            pv: Concerned :class:`ParticleVector`.
            magnitude: Maximum velocity along x.
            wave_number: Number of periods along y.
            sf_channel_name: Name of the channel that will contain the sinusoidal field.
    )");

    m.def("__createStats", &plugin_factory::createStatsPlugin,
          "compute_task"_a, "state"_a, "name"_a, "every"_a, "pvs"_a=std::vector<ParticleVector*>(), "filename"_a="", R"(
        This plugin will report aggregate quantities of all the particles in the simulation:
        total number of particles in the simulation, average temperature and momentum, maximum velocity magnutide of a particle
        and also the mean real time per step in milliseconds.

        .. note::
            This plugin is inactive if postprocess is disabled

        Args:
            name: Name of the plugin.
            every: Report to standard output every that many time-steps.
            pvs: List of pvs to compute statistics from. If empty, will use all the pvs registered in the simulation.
            filename: The statistics are saved in this csv file. The name should either end with `.csv` or have no extension, in which case `.csv` is added.
    )");

    m.def("__createTemperaturize", &plugin_factory::createTemperaturizePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "kBT"_a, "keepVelocity"_a, R"(
        This plugin changes the velocity of each particles from a given :any:`ParticleVector`.
        It can operate under two modes: `keepVelocity = True`, in which case it adds a term drawn from a Maxwell distribution to the current velocity;
        `keepVelocity = False`, in which case it sets the velocity to a term drawn from a Maxwell distribution.

        Args:
            name: name of the plugin
            pv: the concerned :any:`ParticleVector`
            kBT: the target temperature
            keepVelocity: True for adding Maxwell distribution to the previous velocity; False to set the velocity to a Maxwell distribution.
    )");

    m.def("__createVacf", &plugin_factory::createVacfPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "start_time"_a, "end_time"_a, "dump_every"_a, "path"_a, R"(
        This plugin computes the mean velocity autocorrelation over time from a given :any:`ParticleVector`.
        The reference velocity `v0` is that of the given :any:`ParticleVector` at the given start time.

        Args:
            name: Name of the plugin.
            pv: Concerned :class:`ParticleVector`.
            start_time: Simulation time of the reference velocities.
            end_time: End time until which to compute the VACF.
            dump_every: Report the VACF every this many time-steps.
            path: The folder name in which the file will be dumped.
    )");

    m.def("__createVelocityControl", &plugin_factory::createVelocityControlPlugin,
          "compute_task"_a, "state"_a, "name"_a, "filename"_a, "pvs"_a, "low"_a, "high"_a,
          "sample_every"_a, "tune_every"_a, "dump_every"_a, "target_vel"_a, "Kp"_a, "Ki"_a, "Kd"_a, R"(
        This plugin applies a uniform force to all the particles of the target PVS in the specified area (rectangle).
        The force is adapted bvia a PID controller such that the velocity average of the particles matches the target average velocity.

        Args:
            name: Name of the plugin.
            filename: Dump file name. Must have a csv extension or no extension at all.
            pvs: List of concerned :class:`ParticleVector`.
            low, high: boundaries of the domain of interest
            sample_every: sample velocity every this many time-steps
            tune_every: adapt the force every this many time-steps
            dump_every: write files every this many time-steps
            target_vel: the target mean velocity of the particles in the domain of interest
            Kp, Ki, Kd: PID controller coefficients
    )");

    m.def("__createVelocityInlet", &plugin_factory::createVelocityInletPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a,
          "implicit_surface_func"_a, "velocity_field"_a, "resolution"_a, "number_density"_a, "kBT"_a, R"(
        This plugin inserts particles in a given :any:`ParticleVector`.
        The particles are inserted on a given surface with given velocity inlet.
        The rate of insertion is governed by the velocity and the given number density.

        Args:
            name: name of the plugin
            pv: the :any:`ParticleVector` that we ll work with
            implicit_surface_func: a scalar field function that has the required surface as zero level set
            velocity_field: vector field that describes the velocity on the inlet (will be evaluated on the surface only)
            resolution: grid size used to discretize the surface
            number_density: number density of the inserted solvent
            kBT: temperature of the inserted solvent
    )");

    m.def("__createVirialPressurePlugin", &plugin_factory::createVirialPressurePlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "regionFunc"_a, "h"_a, "dump_every"_a, "path"_a, R"(
        This plugin computes the virial pressure from a given :any:`ParticleVector`.
        Note that the stress computation must be enabled with the corresponding stressName.
        This returns the total internal virial part only (no temperature term).
        Note that the volume is not devided in the result, the user is responsible to properly scale the output.

        Args:
            name: name of the plugin
            pv: concerned :class:`ParticleVector`
            regionFunc: predicate for the concerned region; positive inside the region and negative outside
            h: grid size for representing the predicate onto a grid
            dump_every: report total pressure every this many time-steps
            path: the folder name in which the file will be dumped
    )");

    m.def("__createWallRepulsion", &plugin_factory::createWallRepulsionPlugin,
          "compute_task"_a, "state"_a, "name"_a, "pv"_a, "wall"_a, "C"_a, "h"_a, "max_force"_a, R"(
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

        Args:
            name: name of the plugin
            pv: :any:`ParticleVector` that we'll work with
            wall: :any:`Wall` that defines the repulsion
            C: :math:`C`
            h: :math:`h`
            max_force: :math:`F_{max}`
    )");

    m.def("__createWallForceCollector", &plugin_factory::createWallForceCollectorPlugin,
          "compute_task"_a, "state"_a, "name"_a, "wall"_a, "pvFrozen"_a, "sample_every"_a, "dump_every"_a, "filename"_a, "detailed_dump"_a=false, R"(
        This plugin collects and averages the total force exerted on a given wall.
        The result has 2 components:

            * bounce back: force necessary to the momentum change
            * frozen particles: total interaction force exerted on the frozen particles

        Args:
            name: name of the plugin
            wall: The :any:`Wall` to collect forces from
            pvFrozen: corresponding frozen :any:`ParticleVector`
            sample_every: sample every this number of time steps
            dump_every: dump every this number of time steps
            filename: output filename (csv format)
            detailed_dump: if True, will dump separately the bounce contribution and the rest. If False, only the sum is dumped.
    )");
}

} // namespace mirheo
