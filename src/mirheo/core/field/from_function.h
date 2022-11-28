// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "interface.h"

#include <functional>

namespace mirheo {

/// A function that describes a scala field
using ScalarFieldFunction = std::function<real(real3)>;

/** \brief a \c ScalarField that can be initialized from FieldFunction
 */
class ScalarFieldFromFunction : public ScalarField
{
public:
    /** \brief Construct a ScalarFieldFromFunction object
        \param [in] state The global state of the system
        \param [in] name The name of the field object
        \param [in] func The scalar field function
        \param [in] h the grid size
        \param [in] margin Additional margin to store in each rank

        The scalar values will be discretized and stored on the grid.
        This can be useful as one can have a general scalar field configured
        on the host (e.g. from python) but usable on the device.
    */
    ScalarFieldFromFunction(const MirState *state, std::string name,
                            ScalarFieldFunction func, real3 h, real3 margin);
    ~ScalarFieldFromFunction();

    /// move constructor
    ScalarFieldFromFunction(ScalarFieldFromFunction&&);

    void setup(const MPI_Comm& comm) override;

private:
    ScalarFieldFunction func_; ///< The scalar field
};

} // namespace mirheo
