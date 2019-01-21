#include "interface.h"

#include <functional>

using FieldFunction = std::function<float(float3)>;

class FieldFromFunction : public Field
{
public:    
    FieldFromFunction(const YmrState *state, FieldFunction func, float3 h);
    ~FieldFromFunction();

    FieldFromFunction(FieldFromFunction&&);
    
    void setup(MPI_Comm& comm) override;
    
protected:
    
    FieldFunction func;
};
