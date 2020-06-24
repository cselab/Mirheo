#pragma once

#include "interface.h"

#include <string>

namespace mirheo
{

/** \brief Initialize a ParticleVector from a checkpoint file

    Will call the restart() member function of the given ParticleVector.
 */
class RestartIC : public InitialConditions
{
public:
    /** \brief Construct a RestartIC object
        \param [in] path The directory containing the restart files.
     */
    RestartIC(const std::string& path);
    ~RestartIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    std::string path_;
};

} // namespace mirheo
