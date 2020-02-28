#include "interface.h"

#include <functional>

namespace mirheo
{

/// A function that describes a scala field
using FieldFunction = std::function<real(real3)>;

/** \brief a \c Field that can be initialized from FieldFunction
 */
class FieldFromFunction : public Field
{
public:
    /** \brief Construct a FieldFromFunction object
        \param [in] state The global state of the system
        \param [in] name The name of the field object
        \param [in] func The scalar field function
        \param [in] h the grid size

        The scalar values will be discretized and stored on the grid.
        This can be useful as one can have a general scalar field configured 
        on the host (e.g. from python) but usable on the device.
    */
    FieldFromFunction(const MirState *state, std::string name, FieldFunction func, real3 h);
    ~FieldFromFunction();

    /// move constructor
    FieldFromFunction(FieldFromFunction&&);
    
    void setup(const MPI_Comm& comm) override;
    
private:
    FieldFunction func_; ///< The scalar field
};

} // namespace mirheo
