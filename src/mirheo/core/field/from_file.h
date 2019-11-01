#include "interface.h"

namespace mirheo
{

class FieldFromFile : public Field
{
public:    
    FieldFromFile(const MirState *state, std::string name, std::string fieldFileName, real3 h);
    ~FieldFromFile();

    FieldFromFile(FieldFromFile&&);
    
    void setup(const MPI_Comm& comm) override;
    
protected:
    
    std::string fieldFileName;
};

} // namespace mirheo
