// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

#include <vector>

namespace mirheo
{

class ParticleVector;

/** Average particles quantities into spacial bins over a cartesian grid and average it over time.
    Useful to compute e.g. velocity or density profiles.
    The number density is always computed inside each bin, as it is used to compute the averages.
    Other quantities must be specified by giving the channel names.

    This plugin should be used with UniformCartesianDumper on the postprocessing side.

    Cannot be used with multiple invocations of `Mirheo.run`.
 */
class Average3D : public SimulationPlugin
{
public:
    /// Specify the form of a channel data.
    enum class ChannelType : int
    {
     Scalar, Vector_real3, Vector_real4, Tensor6, None
    };

    /// A helper structure that contains grid average info for all required channels.
    struct HostChannelsInfo
    {
        int n; ///< The number of channels (excluding number density).
        std::vector<std::string> names; ///< List of channel names.
        PinnedBuffer<ChannelType> types; ///< List of channel data forms.
        PinnedBuffer<real*> averagePtrs; ///< List of averages of each channel.
        PinnedBuffer<real*> dataPtrs;    ///< List of data to average, for each channel.
        std::vector<DeviceBuffer<real>> average; ///< data container for the averages, for each channel.
    };

    /** Create an \c Average3D object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames The list of names of the ParticleVector that will be used when averaging.
        \param [in] channelNames The list of particle data channels to average. Will die if the channel does not exist.
        \param [in] sampleEvery Compute spatial averages every this number of time steps.
        \param [in] dumpEvery Compute time averages and send to the postprocess side every this number of time steps.
        \param [in] binSize Size of one spatial bin along the three axes.
     */
    Average3D(const MirState *state, std::string name,
              std::vector<std::string> pvNames, std::vector<std::string> channelNames,
              int sampleEvery, int dumpEvery, real3 binSize);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

protected:
    /** Get the number of scalars for a given ChannelType.
        \param [in] type The data form.
        \return Number of scalar values.
    */
    int getNcomponents(ChannelType type) const;

    /** Accumulate all spacial averages into the time average buffers.
        \param [in] stream The compute stream.
     */
    void accumulateSampledAndClear(cudaStream_t stream);

    /** Divide the spacial sums per bin by the number density.
        Divide the number densities by the number of time steps.
        \param [in] stream The compute stream.
    */
    void scaleSampled(cudaStream_t stream);

    /** Perform the binning for all channels on one ParticleVector.
        \param [in] pv The ParticleVector to bin.
        \param [in] stream The compute stream.
     */
    void sampleOnePv(ParticleVector *pv, cudaStream_t stream);

protected:
    std::vector<ParticleVector*> pvs_; ///< list of ParticleVector to collect bin information from.

    int nSamples_ {0}; ///< Current number of per time step samples.
    int sampleEvery_;  ///< Sample every this number of time steps.
    int dumpEvery_;    ///< Send data to dump every this number of time steps.
    int3 resolution_;  ///< Grid size.
    real3 binSize_;    ///< Size of one bin.
    int3 rank3D_;      ///< Rank coordinates in the 3D cartesian communicator.
    int3 nranks3D_;    ///< Number of ranks in the 3D cartesian communicator.

    DeviceBuffer<real>   numberDensity_; ///< Number density for the current sample.
    PinnedBuffer<double> accumulatedNumberDensity_; ///< Sum of number densities, used to average over time.

    HostChannelsInfo channelsInfo_; ///< List of channel average data.
    std::vector<PinnedBuffer<double>> accumulatedAverage_; ///< Accumulated channel samples over time.

    std::vector<char> sendBuffer_; ///< buffer used to communicate with postprocessing side.

private:
    static const std::string numberDensityChannelName_;
    std::vector<std::string> pvNames_;
};

} // namespace mirheo
