#include <mirheo/core/logger.h>
#include "unique_mpi_comm.h"

namespace mirheo
{

void UniqueMPIComm::reset() {
    if (comm_ == MPI_COMM_NULL)
        return;
    int flag;
    MPI_Check( MPI_Finalized(&flag) );
    if (flag) {
        die("Memory leak, an MPI_Comm was not freed before MPI_Finalize. "
            "Maybe the main Mirheo object was destroyed before plugins and "
            "other objects? Maybe mpi4py was destroyed before Mirheo? This "
            "could happen if Mirheo objects are stored in global variables. "
            "Consider also manually deleting the objects.");
    }
    MPI_Check( MPI_Comm_free(&comm_) );  // Sets handle to MPI_COMM_NULL.
}

} // namespace mirheo
