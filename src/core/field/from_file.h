#include "interface.h"

class FieldFromFile : public Field
{
public:    
    FieldFromFile(const YmrState *state, std::string fieldFileName, float3 h);
    ~FieldFromFile();

    FieldFromFile(FieldFromFile&&);
    
    void setup(const MPI_Comm& comm) override;
    
protected:
    
    std::string fieldFileName;
};
