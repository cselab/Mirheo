#include <extern/pybind11/include/pybind11/pybind11.h>

#include <plugins/factory.h>

#include "nodelete.h"

namespace py = pybind11;
using namespace pybind11::literals;

void exportPlugins(py::module& m)
{
    py::class_<SimulationPlugin>  pysim(m, "SimulationPlugin", R"(
        Base simulation plugin class
    )");
    
    py::class_<PostprocessPlugin> pypost(m, "PostprocessPlugin", R"(
        Base postprocess plugin class
    )");
    
    py::nodelete_class<ImposeVelocityPlugin>(m, "ImposeVelocity", pysim);
    py::nodelete_class<TemperaturizePlugin>(m, "Temperaturize", pysim);
    py::nodelete_class<AddForcePlugin>(m, "AddForce", pysim);
    py::nodelete_class<AddTorquePlugin>(m, "AddTorque", pysim);
    py::nodelete_class<ImposeProfilePlugin>(m, "ImposeProfile", pysim);
    py::nodelete_class<WallRepulsionPlugin>(m, "WallRepulsion", pysim);
    
    py::nodelete_class<SimulationStats>(m, "SimulationStats", pysim);
    py::nodelete_class<Average3D>(m, "Average3D", pysim);
    py::nodelete_class<AverageRelative3D>(m, "AverageRelative3D", pysim);
    py::nodelete_class<XYZPlugin>(m, "XYZPlugin", pysim);
    py::nodelete_class<MeshPlugin>(m, "MeshPlugin", pysim);
    py::nodelete_class<ObjPositionsPlugin>(m, "ObjPositions", pysim);
    py::nodelete_class<PinObjectPlugin>(m, "PinObject", pysim);
    
    py::nodelete_class<PostprocessStats>(m, "PostprocessStats", pypost);
    py::nodelete_class<UniformCartesianDumper>(m, "UniformCartesianDumper", pypost);
    py::nodelete_class<XYZDumper>(m, "XYZDumper", pypost);
    py::nodelete_class<MeshDumper>(m, "MeshDumper", pypost);
    py::nodelete_class<ObjPositionsDumper>(m, "ObjPositionsDumper", pypost);
    py::nodelete_class<ReportPinObjectPlugin>(m, "ReportPinObject", pypost);
    
    
    m.def("createImposeVelocity", &PluginFactory::createImposeVelocityPlugin);
    m.def("createTemperaturize", &PluginFactory::createTemperaturizePlugin);
    m.def("createAddForce", &PluginFactory::createAddForcePlugin);
    m.def("createAddTorquePlugin", &PluginFactory::createAddTorquePlugin);
    m.def("createImposeProfilePlugin", &PluginFactory::createImposeProfilePlugin);
    m.def("createWallRepulsionPlugin", &PluginFactory::createWallRepulsionPlugin);
    m.def("createStats", &PluginFactory::createStatsPlugin);
    m.def("createDumpAveragePlugin", &PluginFactory::createDumpAveragePlugin);
    m.def("createDumpAverageRelativePlugin", &PluginFactory::createDumpAverageRelativePlugin);
    m.def("createDumpXYZPlugin", &PluginFactory::createDumpXYZPlugin);
    m.def("createDumpMeshPlugin", &PluginFactory::createDumpMeshPlugin);
    m.def("createDumpObjPosition", &PluginFactory::createDumpObjPosition);
    m.def("createPinObjPlugin", &PluginFactory::createPinObjPlugin);
}

