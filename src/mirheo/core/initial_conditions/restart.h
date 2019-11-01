#pragma once

#include "interface.h"

#include <string>

namespace mirheo
{

class RestartIC : public InitialConditions
{
public:
    RestartIC(std::string path);
    ~RestartIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;
    
private:
    std::string path;
};

} // namespace mirheo
