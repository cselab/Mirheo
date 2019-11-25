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
        

protected:
    std::vector<XDMF::Channel> channels;
    std::unique_ptr<XDMF::UniformGrid> grid;

    std::vector<double> recvNumberDnsity;
    std::vector<std::vector<double>> recvContainers;
    
    std::vector<real> numberDnsity;
    std::vector<std::vector<real>> containers;
    
    std::string path;
    static constexpr int zeroPadding = 5;

    MPI_Comm cartComm {MPI_COMM_NULL};
};

} // namespace mirheo
