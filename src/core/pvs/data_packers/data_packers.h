#pragma once

#include "generic_packer.h"

class ParticlePacker
{
protected:
    GenericPacker particleData;
};

class ObjectPacker : public ParticlePacker
{
protected:
    GenericPacker objectData;;
};

class RodPacker : public ObjectPacker
{
protected:
    GenericPacker bisegmentData;
};
