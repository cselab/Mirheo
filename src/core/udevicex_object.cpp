#include "udevicex_object.h"

UDXObject::UDXObject(std::string name) :
    _name(name)
{}
    
std::string UDXObject::name() const
{
    return _name;
}
