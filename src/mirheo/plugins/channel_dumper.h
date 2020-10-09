// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/unique_mpi_comm.h>
#include <mirheo/core/xdmf/xdmf.h>

#include <memory>
#include <mpi.h>
#include <vector>

namespace mirheo
{

/** Postprocessing side of \c Average3D or \c AverageRelative3D.
    Dump uniform grid data to xmf + hdf5 format.
*/
class UniformCartesianDumper : public PostprocessPlugin
{
public:
    /** Create a UniformCartesianDumper.
        \param [in] name The name of the plugin.
        \param [in] path The files will be dumped to `pathXXXXX.[xmf,h5]`, where `XXXXX` is the time stamp.
     */
    UniformCartesianDumper(std::string name, std::string path);
    ~UniformCartesianDumper();

    void deserialize() override;
    void handshake() override;

    /** Get the average channel data.
        \param [in] chname The name of the channel.
        \return The channel data.
     */
    XDMF::Channel getChannelOrDie(std::string chname) const;

    /** Get the grid size in the local domain.
        \return An array with 3 entries, contains the number of grid points along each direction.
     */
    std::vector<int> getLocalResolution() const;

private:
    std::vector<XDMF::Channel> channels_;
    std::unique_ptr<XDMF::UniformGrid> grid_;

    std::vector<double> recvNumberDnsity_;
    std::vector<std::vector<double>> recvContainers_;

    std::vector<real> numberDnsity_;
    std::vector<std::vector<real>> containers_;

    std::string path_;
    static constexpr int zeroPadding_ = 5;

    UniqueMPIComm cartComm_;
};

} // namespace mirheo
