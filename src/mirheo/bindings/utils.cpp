#include "bindings.h"

#include <mirheo/core/utils/compile_options.h>

#include <pybind11/stl.h>

#include <string>

namespace mirheo
{

using namespace pybind11::literals;

static std::string getCompileOption(const std::string& key)
{
#define check_and_return(opt_name) do {                 \
                  if (key == #opt_name)                 \
                      return CompileOptions::opt_name;  \
              } while(0)

    check_and_return(useNvtx);
    check_and_return(useDouble);
    check_and_return(membraneDouble);
    check_and_return(rodDouble);
    
#undef check_and_return

    throw std::runtime_error("Could not fine the option with name '" + key + "'");
    return "";
}

void exportUtils(py::module& m)
{
    m.def("getCompileOption", getCompileOption, "key"_a R"(
    Fetch a given compile time option currently in use.
    Args:
        key: the option name.

    Available names:
    - useNvtx
    - useDouble
    - membraneDouble
    - rodDouble
    )");
}

} // namespace mirheo
