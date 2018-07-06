#pragma once


template<class Payload>
class PyContainer
{   
protected:
    // Toooooooooo much shit to make it proper unique_ptr
    Payload* impl;
    
public:
    Payload* getImpl() { auto res = impl; impl = nullptr; return res; }
    PyContainer(Payload* ptr) : impl(ptr) {}
};
