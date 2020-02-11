#pragma once

#include <mirheo/core/utils/config.h>

#include <tuple>
#include <typeinfo>

namespace mirheo
{

class Mirheo;
class Mesh;
class ParticleVector;
class Interaction;
class Integrator;

void _unknownRefStringError [[noreturn]] (const std::string &ref);
void _dynamicCastError [[noreturn]] (const char *from, const char *to);

class LoaderContext {
public:
    LoaderContext(std::string snapshotPath);
    LoaderContext(ConfigValue compute, ConfigValue postprocess,
                  std::string snapshotPath = "snapshot/");
    ~LoaderContext();

    const ConfigObject& getCompObjectConfig(const std::string& category,
                                            const std::string& name);

    template <typename T, typename ContainerT = T>
    std::shared_ptr<T> get(const ConfigRefString& ref)
    {
        const auto& container = getContainer<ContainerT>();
        auto it = container.find(parseNameFromRefString(ref));
        if (it == container.end())
            _unknownRefStringError(ref);
        if (T *p = dynamic_cast<T*>(it->second.get()))
            return {it->second, p};
        _dynamicCastError(typeid(it->second.get()).name(), typeid(T).name());
    }

    template <typename T>
    std::map<std::string, std::shared_ptr<T>>& getContainer()
    {
        return std::get<std::map<std::string, std::shared_ptr<T>>>(objects_);
    }

    const std::string& getPath() const { return path_; }
    const ConfigObject& getComp() const { return compConfig_.getObject(); }
    const ConfigObject& getPost() const { return postConfig_.getObject(); }

private:
    template <typename T, typename Factory>
    const std::shared_ptr<T>& _loadObject(Mirheo *mir, Factory factory);

    std::tuple<
        std::map<std::string, std::shared_ptr<Mesh>>,
        std::map<std::string, std::shared_ptr<ParticleVector>>,
        std::map<std::string, std::shared_ptr<Interaction>>,
        std::map<std::string, std::shared_ptr<Integrator>>> objects_;
    std::string path_;
    ConfigValue compConfig_;
    ConfigValue postConfig_;
};

void loadSnapshot(Mirheo *mir, Loader& loader);

} // namespace mirheo
