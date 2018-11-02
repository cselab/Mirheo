#include <string>

class UDXObject
{
public:

    UDXObject(std::string name);

    virtual ~UDXObject() = default;
    
    std::string name() const;
    
private:
    std::string _name;
};
