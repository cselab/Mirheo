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

    void deserialize(MPI_Status& stat) override;
    void handshake() override;
        

protected:
    std::vector<XDMF::Channel> channels;
    std::unique_ptr<XDMF::UniformGrid> grid;
    
    std::vector<float> density;
    std::vector<std::vector<float>> containers;
    
    std::string path;
    int timeStamp = 0;
    const int zeroPadding = 5;

    MPI_Comm cartComm;
};
