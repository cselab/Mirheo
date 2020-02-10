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
    std::shared_ptr<T> getShared(const ConfigRefString& ref)
    {
        const auto& container = getContainerShared<ContainerT>();
        auto it = container.find(parseNameFromRefString(ref));
        if (it == container.end())
            _unknownRefStringError(ref);
        if (T *p = dynamic_cast<T*>(it->second.get()))
            return {it->second, p};
        _dynamicCastError(typeid(it->second.get()).name(), typeid(T).name());
    }

    template <typename T>
    std::unique_ptr<T> getUnique(const ConfigRefString& ref)
    {
        if (ref == ConfigNullRefString)
            return nullptr;
        auto& container = getContainerUnique<T>();
        auto it = container.find(parseNameFromRefString(ref));
        if (it == container.end())
            _unknownRefStringError(ref);
        return std::move(it->second);
    }

    template <typename T>
    std::map<std::string, std::shared_ptr<T>>& getContainerShared()
    {
        return std::get<std::map<std::string, std::shared_ptr<T>>>(shared_);
    }
    template <typename T>
    std::map<std::string, std::unique_ptr<T>>& getContainerUnique()
    {
        return std::get<std::map<std::string, std::unique_ptr<T>>>(unique_);
    }

    const std::string& getPath() const { return path_; }
    const ConfigObject& getComp() const { return compConfig_.getObject(); }
    const ConfigObject& getPost() const { return postConfig_.getObject(); }

private:

    std::tuple<
        std::map<std::string, std::shared_ptr<Mesh>>,
        std::map<std::string, std::shared_ptr<ParticleVector>>,
        std::map<std::string, std::shared_ptr<Interaction>>> shared_;
    std::tuple<
        std::map<std::string, std::unique_ptr<Interaction>>> unique_;
    std::string path_;
    ConfigValue compConfig_;
    ConfigValue postConfig_;
};

void loadSnapshot(Mirheo *mir, Loader& loader);

} // namespace mirheo
