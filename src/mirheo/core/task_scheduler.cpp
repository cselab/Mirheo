// Copyright 2020 ETH Zurich. All Rights Reserved.
#include <mirheo/core/task_scheduler.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/nvtx.h>

#include <extern/pugixml/src/pugixml.hpp>

#include <algorithm>
#include <fstream>
#include <memory>
#include <queue>
#include <sstream>
#include <unistd.h>

namespace mirheo
{

TaskScheduler::TaskScheduler()
{
    CUDA_Check( cudaDeviceGetStreamPriorityRange(&cudaPriorityLow_, &cudaPriorityHigh_) );
}

TaskScheduler::~TaskScheduler()
{
    auto destroyStreams = [](std::queue<cudaStream_t>& streams)
    {
        while (!streams.empty())
        {
            CUDA_Check( cudaStreamDestroy(streams.front()) );
            streams.pop();
        }
    };

    destroyStreams(streamsLo_);
    destroyStreams(streamsHi_);
}

TaskScheduler::TaskID TaskScheduler::createTask(const std::string& label)
{
    auto id = getTaskId(label);
    if (id != invalidTaskId)
        die("Task '%s' already exists", label.c_str());

    id = static_cast<int>(tasks_.size());
    label2taskId_[label] = id;

    Task task(label, id, cudaPriorityLow_);
    tasks_.push_back(task);

    return id;
}

TaskScheduler::TaskID TaskScheduler::getTaskId(const std::string& label)
{
    if (label2taskId_.find(label) != label2taskId_.end())
        return label2taskId_[label];
    else
        return invalidTaskId;
}

TaskScheduler::TaskID TaskScheduler::getTaskIdOrDie(const std::string& label)
{
    auto id = getTaskId(label);
    if (id == invalidTaskId)
        die("No such task '%s'", label.c_str());

    return id;
}

void TaskScheduler::_checkTaskExistsOrDie(TaskID id) const
{
    if (id >= static_cast<TaskID>(tasks_.size()) || id < 0)
        die("No such task with id %d", id);
}

TaskScheduler::Node* TaskScheduler::_getNode(TaskID id)
{
    for (auto& n : nodes_)
        if (n->id == id) return n.get();

    return nullptr;
}

TaskScheduler::Node* TaskScheduler::_getNodeOrDie(TaskID id)
{
    auto node = _getNode(id);
    if (node == nullptr)
        die("No such task with id %d", id);

    return node;
}


void TaskScheduler::addTask(TaskID id, TaskScheduler::Function task, int every)
{
    _checkTaskExistsOrDie(id);

    if (every <= 0)
        die("'every' must be non negative: got %d???", every);

    tasks_[id].funcs.push_back({task, every});
}


void TaskScheduler::addDependency(TaskID id, std::vector<TaskID> before, std::vector<TaskID> after)
{
    _checkTaskExistsOrDie(id);
    tasks_[id].before.insert(tasks_[id].before.end(), before.begin(), before.end());
    tasks_[id].after .insert(tasks_[id].after .end(), after .begin(), after .end());
}

void TaskScheduler::setHighPriority(TaskID id)
{
    _checkTaskExistsOrDie(id);
    tasks_[id].priority = cudaPriorityHigh_;
}

void TaskScheduler::forceExec(TaskID id, cudaStream_t stream)
{
    _checkTaskExistsOrDie(id);
    debug("Forced execution of group %s", tasks_[id].label.c_str());

    for (auto& func_every : tasks_[id].funcs)
        func_every.first(stream);
}


void TaskScheduler::_createNodes()
{
    nodes_.clear();

    for (auto& t : tasks_)
    {
        auto node = std::make_unique<Node>(t.id, t.priority);
        nodes_.push_back(std::move(node));
    }

    for (auto& n : nodes_)
    {
        // Set streams member according to priority
        if      (n->priority == cudaPriorityLow_)
            n->streams = &streamsLo_;
        else if (n->priority == cudaPriorityHigh_)
            n->streams = &streamsHi_;
        else
            n->streams = &streamsLo_;

        // Set dependencies
        for (auto dep : tasks_[n->id].before)
        {
            Node* depPtr = _getNode(dep);

            if (depPtr == nullptr)
            {
                error("Could not resolve dependency %s  -->  id %d, trying to move on", tasks_[n->id].label.c_str(), dep);
                continue;
            }

            n->to.push_back(depPtr);
            depPtr->from_backup.push_back(n.get());
        }

        for (auto dep : tasks_[n->id].after)
        {
            Node* depPtr = _getNode(dep);

            if (depPtr == nullptr)
            {
                error("Could not resolve dependency id %d  -->  %s, trying to move on", dep, tasks_[n->id].label.c_str());
                continue;
            }

            n->from_backup.push_back(depPtr);
            depPtr->to.push_back(n.get());
        }
    }
}


void TaskScheduler::_removeEmptyNodes()
{
    for (auto it = nodes_.begin(); it != nodes_.end(); )
    {
        auto checkedNode = it->get();

        if ( tasks_[checkedNode->id].funcs.size() == 0 )
        {
            debug("Task '%s' is empty and will be removed from execution", tasks_[checkedNode->id].label.c_str());
            for (auto& n : nodes_)
            {
                const auto toSize = n->to.size();
                const auto from_backupSize = n->from_backup.size();

                // Others cannot have dependencies with the removed
                n->to.remove(checkedNode);
                n->from_backup.remove(checkedNode);

                // If some arrows were removed, add the deps from removed node
                if (toSize != n->to.size())
                    n->to.insert( n->to.end(), checkedNode->to.begin(), checkedNode->to.end() );

                if (from_backupSize != n->from_backup.size())
                    n->from_backup.insert( n->from_backup.end(), checkedNode->from_backup.begin(), checkedNode->from_backup.end() );

                // Add deps from the removed node to all that it depends on/off
                if ( std::find(checkedNode->to.begin(), checkedNode->to.end(), n.get()) != checkedNode->to.end() )
                    n->from_backup.insert( n->from_backup.end(), checkedNode->from_backup.begin(), checkedNode->from_backup.end() );

                if ( std::find(checkedNode->from_backup.begin(), checkedNode->from_backup.end(), n.get()) != checkedNode->from_backup.end() )
                    n->to.insert( n->to.end(), checkedNode->to.begin(), checkedNode->to.end() );
            }

            it = nodes_.erase(it);
        }
        else
            it++;
    }

    // Cleanup dependencies
    for (auto& n : nodes_)
    {
        n->from_backup.sort();
        n->from_backup.unique();

        n->to.sort();
        n->to.unique();
    }
}

void TaskScheduler::_logDepsGraph()
{
    debug("Task graph consists of total %zu tasks:", nodes_.size());

    for (auto& n : nodes_)
    {
        std::stringstream str;

        auto& task = tasks_[n->id];
        str << "Task '" << task.label << "', id " << n->id << " with " << task.funcs.size() << " functions" << std::endl;

        if (n->to.size() > 0)
        {
            str << "    Before tasks:" << std::endl;
            for (auto dep : n->to)
                str << "     * " << tasks_[dep->id].label << std::endl;
        }

        if (n->from_backup.size() > 0)
        {
            str << "    After tasks:" << std::endl;
            for (auto dep : n->from_backup)
                str << "     * " << tasks_[dep->id].label << std::endl;
        }

        debug("%s", str.str().c_str());
    }
}

void TaskScheduler::compile()
{
    _createNodes();
    _removeEmptyNodes();
    _logDepsGraph();
}



void TaskScheduler::run()
{
    // Kahn's algorithm
    // https://en.wikipedia.org/wiki/Topological_sorting

    auto compareNodes = [] (Node *a, Node *b) {
        // lower number means higher priority
        return a->priority < b->priority;
    };
    std::priority_queue<Node*, std::vector<Node*>, decltype(compareNodes)> S(compareNodes);
    std::vector<std::pair<cudaStream_t, Node*>> workMap;

    for (auto& n : nodes_)
    {
        n->from = n->from_backup;

        if (n->from.empty())
            S.push(n.get());
    }

    int completed = 0;
    const int total = static_cast<int>(nodes_.size());

    while (true)
    {
        // Check the status of all running kernels
        while (completed < total && S.empty())
        {
            for (auto streamNode_it = workMap.begin(); streamNode_it != workMap.end(); )
            {
                auto result = cudaStreamQuery(streamNode_it->first);
                if ( result == cudaSuccess )
                {
                    auto node = streamNode_it->second;

                    debug("Completed group %s ", tasks_[node->id].label.c_str());

                    // Return freed stream back to the corresponding queue
                    node->streams->push(streamNode_it->first);

                    // Remove resolved dependencies
                    for (auto dep : node->to)
                    {
                        if (!dep->from.empty())
                        {
                            dep->from.remove(node);
                            if (dep->from.empty())
                                S.push(dep);
                        }
                    }

                    // Remove task from the list of currently in progress
                    completed++;
                    streamNode_it = workMap.erase(streamNode_it);
                }
                else if (result == cudaErrorNotReady)
                {
                    streamNode_it++;
                }
                else
                {
                    error("Group '%s' raised an error",  tasks_[streamNode_it->second->id].label.c_str());
                    CUDA_Check( result );
                }
            }
        }

        if (completed == total)
            break;

        Node* node = S.top();
        S.pop();

        cudaStream_t stream;
        if (node->streams->empty())
        {
            CUDA_Check( cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, node->priority) );
        }
        else
        {
            stream = node->streams->front();
            node->streams->pop();
        }

        debug("Executing group %s on stream %lld with priority %d",
              tasks_[node->id].label.c_str(), (long long)stream, node->priority);
        workMap.push_back({stream, node});

        {
            auto& task = tasks_[node->id];
            NvtxCreateRange(range, task.label.c_str());

            for (auto& func_every : task.funcs)
                if (nExecutions_ % func_every.second == 0)
                    func_every.first(stream);
        }
    }

    nExecutions_++;
    CUDA_Check( cudaDeviceSynchronize() );
}


static void add_node(pugi::xml_node& graph, int id, std::string label)
{
    auto node = graph.append_child("node");
    node.append_attribute("id") = std::to_string(id).c_str();

    auto data = node.append_child("data");
    data.append_attribute("key") = "label";
    data.text()                  = label.c_str();
}

static void add_edge(pugi::xml_node& graph, int sourceId, int targetId)
{
    auto edge = graph.append_child("edge");
    edge.append_attribute("source") = std::to_string(sourceId).c_str();
    edge.append_attribute("target") = std::to_string(targetId).c_str();
}

void TaskScheduler::dumpGraphToGraphML(const std::string& fname) const
{
    pugi::xml_document doc;
    auto root = doc.append_child("graphml");

    root.append_attribute("xmlns")              = "http://graphml.graphdrawing.org/xmlns";
    root.append_attribute("xmlns:xsi")          = "http://www.w3.org/2001/XMLSchema-instance";
    root.append_attribute("xsi:schemaLocation") = "http://graphml.graphdrawing.org/xmlns "
                                                  "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd";

    auto key = root.append_child("key");
    key.append_attribute("id")        = "label";
    key.append_attribute("for")       = "node";
    key.append_attribute("attr.name") = "label";
    key.append_attribute("attr.type") = "string";

    auto graph = root.append_child("graph");
    graph.append_attribute("id")          = "Task graph";
    graph.append_attribute("edgedefault") = "directed";

    // Nodes
    for (const auto& t : tasks_)
        add_node(graph, t.id, t.label);

    // Edges
    for (const auto& n : nodes_)
    {
        for (auto dep : n->to)
            add_edge(graph, n->id, dep->id);

        for (auto dep : n->from_backup)
            add_edge(graph, dep->id, n->id);
    }

    auto filename = fname + ".graphml";
    doc.save_file(filename.c_str());
}

TaskScheduler::Task::Task(const std::string& label_, TaskID id_, int priority_) :
    label(label_),
    id(id_),
    priority(priority_)
{}

TaskScheduler::Node::Node(TaskID id_, int priority_) :
    id(id_),
    priority(priority_)
{}

} // namespace mirheo
