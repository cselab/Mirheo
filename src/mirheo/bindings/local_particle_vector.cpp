// Copyright 2021 ETH Zurich. All Rights Reserved.

#include "bindings.h"
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
                return channel ? channel : throw py::key_error(name);
            }, "name"_a, R"(
                Returns:
                    The ChannelDescription for the given channel name.
            )", py::return_value_policy::reference_internal);

    pylpv.def("__iter__", [](LocalParticleVector *lpv)
            {
                // getSortedChannels returns const pointers, but pybind11 strips
                // constness away (which is desired here).
                const auto &channels = lpv->dataPerParticle.getSortedChannels();
                return py::make_key_iterator(channels.begin(), channels.end());
            }, py::keep_alive<0, 1>());

    pylpv.def("items", [](LocalParticleVector *lpv)
            {
                const auto &channels = lpv->dataPerParticle.getSortedChannels();
                return py::make_iterator(channels.begin(), channels.end());
            }, py::keep_alive<0, 1>());
}

} // namespace mirheo
