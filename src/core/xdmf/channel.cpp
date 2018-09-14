#include "channel.h"

#include <core/logger.h>

namespace XDMF
{
    Channel::Channel(std::string name, void* data, Type type, int entrySize_bytes, std::string typeStr) :
        name(name), data((float*)data), type(type),
        entrySize_floats(entrySize_bytes / sizeof(float)), typeStr(typeStr)
    {
        if (entrySize_floats*sizeof(float) != entrySize_bytes)
            die("Channel('%s') should have a chunk size in bytes divisible by %d (got %d)",
                name.c_str(), sizeof(float), entrySize_bytes);
    }
}
