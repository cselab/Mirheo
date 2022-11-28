// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "interface.h"

namespace mirheo {

/** \brief a \c ScalarField that can be initialized from a file
 */
class ScalarFieldFromFile : public ScalarField
{
public:
    /** \brief Construct a ScalarFieldFromFile object

        \param [in] state The global state of the system
        \param [in] name The name of the field object
        \param [in] fieldFileName The input file name
        \param [in] h the grid size
        \param [in] margin Additional margin to store in each rank

        The format of the file is custom.
        It is a single file that contains a header followed by the data grid data in binary format.
        The header is composed of two lines in ASCII format:
        - domain size (3 floating point numbers)
        - number of grid points (3 integers)

        The data is an array that contains all grid values (x is the fast running index).
    */
    ScalarFieldFromFile(const MirState *state, std::string name,
                  std::string fieldFileName, real3 h, real3 margin);
    ~ScalarFieldFromFile();

    /// move constructor
    ScalarFieldFromFile(ScalarFieldFromFile&&);

    void setup(const MPI_Comm& comm) override;

private:
    std::string fieldFileName_; ///< file name
};

} // namespace mirheo
