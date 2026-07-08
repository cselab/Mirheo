// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>

#include <memory>
#include <string>

namespace mirheo
{

namespace rdf_plugin
{
using ReductionType = double;
using CountType = unsigned long long;
}

class CellList;
class ParticleVector;

/** Measure the radial distribution function (RDF) of a ParticleVector.
    The RDF is estimated periodically from ensemble averages.
    See RdfDump for the I/O.
*/
class RdfPlugin : public SimulationPlugin
{
public:
    /** Create a RdfPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector from which to measure the RDF.
        \param [in] maxDist The RDF will be measured on [0, maxDist].
        \param [in] nbins The number of bins in the interval [0, maxDist].
        \param [in] computeEvery The number of time steps between two RDF evaluations and dump.
    */
    RdfPlugin(const MirState *state, std::string name, std::string pvName, real maxDist, int nbins, int computeEvery);

    ~RdfPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    std::string pvName_;
    real maxDist_;
    int nbins_;
    int computeEvery_;

    bool needToDump_{false};
    ParticleVector *pv_;

    PinnedBuffer<rdf_plugin::CountType> nparticles_{1};
    PinnedBuffer<rdf_plugin::CountType> countsPerBin_;

    std::vector<char> sendBuffer_;
    std::unique_ptr<CellList> cl_;
};

/** Postprocess side of RdfPlugin.
    Dump the RDF to a csv file.
*/
class RdfDump : public PostprocessPlugin
{
public:
    /** Create a RdfDump object.
        \param [in] name The name of the plugin.
        \param [in] basename The RDF will be dumped to `basenameXXXXX.csv`.
    */
    RdfDump(std::string name, std::string basename);

    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void deserialize() override;

private:
    std::string basename_;
};

} // namespace mirheo
