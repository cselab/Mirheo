#include "udevicex_object.h"

UdxObject::UdxObject(std::string name) :
    _name(name)
{}
    
const std::string& UdxObject::name() const
{
    return _name;
}
