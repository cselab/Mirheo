#pragma once

#include <mpi.h>

namespace mirheo
{

/** A std::unique_ptr-like wrapper of MPI_Comm.

    Automatically frees the communicator in the destructor, while checking
    whether the MPI has already been finalized or not.
*/
class UniqueMPIComm
{
public:
    /// Default constructor. Initialize to \c MPI_COMM_NULL.
    UniqueMPIComm() noexcept : comm_{MPI_COMM_NULL} { }

    UniqueMPIComm(const UniqueMPIComm &) = delete;

    /** Move constructor.
        Set the other to \c MPI_COMM_NULL.
        \param other Th UniqueMPIComm to move from.
    */
    UniqueMPIComm(UniqueMPIComm &&other) : comm_{other.comm_}
    {
        other.comm_ = MPI_COMM_NULL;
    }

    UniqueMPIComm& operator=(const UniqueMPIComm &) = delete;

    /** Move assignement.
        \param other Th UniqueMPIComm to move from.
     */
    UniqueMPIComm& operator=(UniqueMPIComm &&other)
    {
        reset();
        comm_ = other.comm_;
        other.comm_ = MPI_COMM_NULL;
        return *this;
    }

    ~UniqueMPIComm() { reset(); }

    /// Deallocate the stored MPI communicator, if any.
    void reset();

    /** Deallocate the stored MPI communicator, if any, and return the address
        of the internal MPI_Comm. Useful for MPI APIs which create new communicators.

        Example:
            UniqueMPIComm comm;
            MPI_Check( MPI_Comm_dup(oldComm, comm.reset_and_get_adress()) );
    */
    MPI_Comm *reset_and_get_address() {
        reset();
        return &comm_;
    }

    /// \return MPI_Comm
    operator MPI_Comm() { return comm_; }
private:
    MPI_Comm comm_;
};

} // namespace mirheo
