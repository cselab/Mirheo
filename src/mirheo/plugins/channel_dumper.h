// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>
#include <mirheo/core/xdmf/xdmf.h>

#include <memory>
#include <mpi.h>
#include <vector>

namespace mirheo
{

class UniformCartesianDumper : public PostprocessPlugin
{
public:
    UniformCartesianDumper(std::string name, std::string path);
    ~UniformCartesianDumper();

    void deserialize() override;
    void handshake() override;

    XDMF::Channel getChannelOrDie(std::string chname) const;
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

    MPI_Comm cartComm_ {MPI_COMM_NULL};
};

} // namespace mirheo
