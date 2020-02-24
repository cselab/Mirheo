#include "bindings.h"

#include <mirheo/core/utils/compile_options.h>

#include <pybind11/stl.h>

#include <string>

namespace mirheo
{

using namespace pybind11::literals;

static std::string getCompileOption(const std::string& key)
{
#define check_and_return(opt_name)                                      \
    if (key == #opt_name)                                               \
        return CompileOptions::opt_name;
    
    MIRHEO_COMPILE_OPT_TABLE(check_and_return);
    
#undef check_and_return

    throw std::runtime_error("Could not fine the option with name '" + key + "'");
    return "";
}

static std::map<std::string, std::string> getAllCompileOptions()
{
    std::map<std::string, std::string> dict;
    
#define add_option(opt_name) dict[#opt_name] = CompileOptions::opt_name;
    MIRHEO_COMPILE_OPT_TABLE(add_option);
#undef add_option

    return dict;
}

void exportUtils(py::module& m)
{
    m.def("getCompileOption", getCompileOption, "key"_a, R"(
    Fetch a given compile time option currently in use.
    Args:
        key: the option name.

    Available names can be found from the :any:`getAllCompileOptions` command.
    )");

    m.def("getAllCompileOptions", getAllCompileOptions, R"(
    Return all compile time options used in the current installation in the form of a dictionary.
    )");
}

} // namespace mirheo
