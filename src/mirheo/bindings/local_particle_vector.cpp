// Copyright 2021 ETH Zurich. All Rights Reserved.

#include "bindings.h"
#include "cuda_array_interface.h"
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

using namespace py::literals;

void exportLocalParticleVector(py::module& m) {
    py::class_<LocalParticleVector> pylpv(m, "LocalParticleVector", R"(
        Particle data storage, a collection of channels.
    )");

    pylpv.def("__getitem__", [](LocalParticleVector *lpv, const std::string &name)
            {
                auto * const channel = lpv->dataPerParticle.getChannelDesc(name);
                if (channel == nullptr)
                    throw py::key_error(name);
                return getVariantCudaArrayInterface(channel->varDataPtr);
            }, "name"_a, py::keep_alive<0, 1>(), R"(
                Returns:
                    Cupy-compatible view over the internal CUDA buffer.
            )");
    pylpv.def("__iter__", [](LocalParticleVector *lpv)
            {
                const auto &channels = lpv->dataPerParticle.getSortedChannels();
                return py::make_key_iterator(channels.begin(), channels.end());
            }, py::keep_alive<0, 1>(), R"(
                Returns:
                    Iterator over channel names.
            )");

    pylpv.def_property_readonly("r", [](LocalParticleVector& lpv)
            {
                CudaArrayInterface array = getBufferCudaArrayInterface(lpv.positions());
                assert(array.shape[1] == 4);
                array.shape[1] = 3;
                return array;
            }, py::keep_alive<0, 1>(), R"(
                Alias for the `real3` part of `lpv['positions']`.

                Returns:
                    Cupy-compatible view over the positions buffer.
            )");
    pylpv.def_property_readonly("v", [](LocalParticleVector& lpv)
            {
                CudaArrayInterface array = getBufferCudaArrayInterface(lpv.velocities());
                assert(array.shape[1] == 4);
                array.shape[1] = 3;
                return array;
            }, py::keep_alive<0, 1>(), R"(
                Alias for the `real3` part of `lpv['velocities']`.

                Returns:
                    Cupy-compatible view over the velocities buffer.
            )");
    pylpv.def_property_readonly("f", [](LocalParticleVector& lpv)
            {
                // Here the `int` part of the `Force` struct is already stripped away.
                return getBufferCudaArrayInterface(lpv.forces());
            }, py::keep_alive<0, 1>(), R"(
                Alias for the `real3` part of `lpv['__forces']`.

                Returns:
                    Cupy-compatible view over the forces buffer.
            )");
}

} // namespace mirheo
