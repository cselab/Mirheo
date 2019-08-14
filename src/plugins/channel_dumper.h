#pragma once

#include <vector>
#include <memory>
#include <mpi.h>

#include <plugins/interface.h>
#include <core/xdmf/xdmf.h>

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

    std::vector<double> recv_density;
    std::vector<std::vector<double>> recv_containers;
    
    std::vector<float> density;
    std::vector<std::vector<float>> containers;
    
    std::string path;
    const int zeroPadding = 5;

    MPI_Comm cartComm {MPI_COMM_NULL};
};
