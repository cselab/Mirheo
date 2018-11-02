#pragma once

#include <string>

class UdxObject
{
public:

    UdxObject(std::string name);

    virtual ~UdxObject() = default;
    
    std::string name() const;
    
private:
    std::string _name;
};
