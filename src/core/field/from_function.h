#include "interface.h"

#include <functional>

using FieldFunction = std::function<real(real3)>;

class FieldFromFunction : public Field
{
public:    
    FieldFromFunction(const MirState *state, std::string name, FieldFunction func, real3 h);
    ~FieldFromFunction();

    FieldFromFunction(FieldFromFunction&&);
    
    void setup(const MPI_Comm& comm) override;
    
protected:
    
    FieldFunction func;
};
