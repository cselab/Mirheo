#include "interface.h"

namespace mirheo
{
/** \brief a \c Field that can be initialized from a file
 */
class FieldFromFile : public Field
{
public:
    /** \brief Construct a FieldFromFile object

        \param [in] state The global state of the system
        \param [in] name The name of the field object
        \param [in] fieldFileName The input file name
        \param [in] h the grid size

        The format of the file is custom.
        It is a single file that contains a header followed by the data grid data in binary format.
        The header is composed of two lines in ASCII format: 
        - domain size (3 floating point numbers)
        - number of grid points (3 integers)

        The data is an array that contains all grid values (x is the fast running index).
    */
    FieldFromFile(const MirState *state, std::string name, std::string fieldFileName, real3 h);
    ~FieldFromFile();

    /// move constructor
    FieldFromFile(FieldFromFile&&);
    
    void setup(const MPI_Comm& comm) override;
    
private:
    std::string fieldFileName_; ///< file name
};

} // namespace mirheo
