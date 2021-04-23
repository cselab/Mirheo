// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/utils/config.h>

#include <mpi.h>
#include <tuple>
#include <typeinfo>

namespace mirheo
{

class Integrator;
class Interaction;
class Mesh;
class MirState;
class Mirheo;
class ObjectBelongingChecker;
class ParticleVector;
class PostprocessPlugin;
class SimulationPlugin;
class Wall;

/// Report an error that a refstring was not found. Abort.
void _unknownRefStringError [[noreturn]] (const std::string &ref);

/// Report an error of a failed dynamic cast. Abort.
void _dynamicCastError [[noreturn]] (const char *from, const char *to);

/// Snapshot parameters and temporary variables.
class SaverContext
{
public:
    std::string path {"snapshot/"};      ///< Snapshot folder path.
    MPI_Comm groupComm {MPI_COMM_NULL};  ///< Current's rank group communicator (compute or postprocessing).
    std::map<std::string, int> counters; ///< Map of named counters.

    /// Returns true if the current rank is a master compute or master postprocessing rank.
    bool isGroupMasterTask() const;
};

/** \brief Helper class for snapshot loading.

    Stores loading context (snapshot path and config JSON), as well as the
    loaded objects.
 */
class LoaderContext {
public:
    /// Construct a load context from a given snapshot path.
    LoaderContext(std::string snapshotPath);

    /** \brief Construct a load context from a manually given config object.

        Useful for altering the config object before it is loaded.

        \param config A config object, metadata of a snapshot. Potentially different from config.json.
        \param snapshotPath The path to the snapshot folder.
     */
    LoaderContext(ConfigValue config, std::string snapshotPath = "snapshot/");

    ~LoaderContext();

    /** \brief Get a pointer to the object referred by the given refstring.

        \tparam T Target pointer type.
        \tparam ContainerT Stored pointer type.
        \param ref Refstring of the object.

        If T and ContainerT differ, a dynamic cast will be performed.
     */
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

    /// Get the container (map) storing objects of the given category.
    template <typename T>
    std::map<std::string, std::shared_ptr<T>>& getContainer()
    {
        return std::get<std::map<std::string, std::shared_ptr<T>>>(objects_);
    }

    /// Return the snapshot folder path.
    const std::string& getPath() const noexcept { return path_; }

    /// Return the config object.
    const ConfigObject& getConfig() const { return config_.getObject(); }

private:
    template <typename T, typename Factory>
    const std::shared_ptr<T>& _loadObject(Mirheo *mir, Factory factory);

    // TODO: Is this separation of types even necessary, shouldn't an
    // std::map<std::string, std::shared_ptr<MirObject>> be enough?
    std::tuple<
        std::map<std::string, std::shared_ptr<Mesh>>,
        std::map<std::string, std::shared_ptr<ObjectBelongingChecker>>,
        std::map<std::string, std::shared_ptr<ParticleVector>>,
        std::map<std::string, std::shared_ptr<Interaction>>,
        std::map<std::string, std::shared_ptr<Integrator>>,
        std::map<std::string, std::shared_ptr<Wall>>> objects_;
    std::string path_;
    ConfigValue config_;
};

/** This is a mechanism for avoiding undefined symbols during the linking phase
    since the Mirheo core is compiled independently from plugins. In other
    words, since plugins are treated as optional, this is a mechanism to add
    factory for loading plugin snapshots.

    First, each plugin set (such as the one provided with Mirheo) registers its
    factory. Then, during the snapshot loading, for each pair of stored
    compute/postprocess plugins, the `loadSnapshot` function will traverse
    every factory until one of them successfully constructs the plugins. If
    none of them do, an exception will be thrown.
 */
class PluginFactoryContainer
{
public:
    /** \brief Return value of a plugin factory, a tuple (found, simPlugin, postPlugin).

        Note that a pair of shared pointers does not suffice, since on
        postprocess ranks we would not be able to differentiate between
        simulation-only plugins and unrecognized plugins: both would return a
        {nullptr, nullptr} pair.
     */
    using OptionalPluginPair = std::tuple<
        bool,
        std::shared_ptr<SimulationPlugin>,
        std::shared_ptr<PostprocessPlugin>>;

    /// Factory type. The factory receives the MirState object, loader, and at
    /// least one of the simulation and postprocess plugin configs.
    /// Note: Can be changed to std::function if needed.
    using FactoryType = OptionalPluginPair(*)(
            bool computeTask, const MirState *, Loader&,
            const ConfigObject *sim, const ConfigObject *post);

    /// Get singleton.
    static PluginFactoryContainer& get() noexcept;

    /// Register the factory.
    void registerPluginFactory(FactoryType factory);

    /// Getter for the vector of factories.
    const std::vector<FactoryType>& getFactories() const noexcept {
        return factories_;
    }
private:
    std::vector<FactoryType> factories_;
};

/// Load the snapshot to the Mirheo object.
void loadSnapshot(Mirheo *mir, Loader& loader);

/// Create a snapshot path from the prefix (pattern) and the snapshot ID.
std::string createSnapshotPath(const std::string& pathPrefix, int snapshotId);

} // namespace mirheo
