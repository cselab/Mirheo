#include <mirheo/core/task_scheduler.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/nvtx.h>

#include <extern/pugixml/src/pugixml.hpp>

#include <fstream>
#include <memory>
#include <queue>
#include <sstream>
#include <unistd.h>

TaskScheduler::TaskScheduler()
{
    CUDA_Check( cudaDeviceGetStreamPriorityRange(&cudaPriorityLow, &cudaPriorityHigh) );
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

    destroyStreams(streamsLo);
    destroyStreams(streamsHi);
}

TaskScheduler::TaskID TaskScheduler::createTask(const std::string& label)
{
    auto id = getTaskId(label);
    if (id != invalidTaskId)
        die("Task '%s' already exists", label.c_str());

    id = tasks.size();
    label2taskId[label] = id;

    Task task(label, id, cudaPriorityLow);
    tasks.push_back(task);

    return id;
}

TaskScheduler::TaskID TaskScheduler::getTaskId(const std::string& label)
{
    if (label2taskId.find(label) != label2taskId.end())
        return label2taskId[label];
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

void TaskScheduler::checkTaskExistsOrDie(TaskID id) const
{
    if (id >= static_cast<TaskID>(tasks.size()) || id < 0)
        die("No such task with id %d", id);
}

TaskScheduler::Node* TaskScheduler::getNode(TaskID id)
{
    for (auto& n : nodes)
        if (n->id == id) return n.get();

    return nullptr;
}

TaskScheduler::Node* TaskScheduler::getNodeOrDie(TaskID id)
{
    auto node = getNode(id);
    if (node == nullptr)
        die("No such task with id %d", id);

    return node;
}


void TaskScheduler::addTask(TaskID id, TaskScheduler::Function task, int every)
{
    checkTaskExistsOrDie(id);

    if (every <= 0)
        die("'every' must be non negative: got %d???", every);

    tasks[id].funcs.push_back({task, every});
}


void TaskScheduler::addDependency(TaskID id, std::vector<TaskID> before, std::vector<TaskID> after)
{
    checkTaskExistsOrDie(id);
    tasks[id].before.insert(tasks[id].before.end(), before.begin(), before.end());
    tasks[id].after .insert(tasks[id].after .end(), after .begin(), after .end());
}

void TaskScheduler::setHighPriority(TaskID id)
{
    checkTaskExistsOrDie(id);
    tasks[id].priority = cudaPriorityHigh;
}

void TaskScheduler::forceExec(TaskID id, cudaStream_t stream)
{
    checkTaskExistsOrDie(id);
    debug("Forced execution of group %s", tasks[id].label.c_str());

    for (auto& func_every : tasks[id].funcs)
        func_every.first(stream);
}


void TaskScheduler::createNodes()
{
    nodes.clear();

    for (auto& t : tasks)
    {
        auto node = std::make_unique<Node>(t.id, t.priority);
        nodes.push_back(std::move(node));
    }

    for (auto& n : nodes)
    {
        // Set streams member according to priority
        if      (n->priority == cudaPriorityLow)
            n->streams = &streamsLo;
        else if (n->priority == cudaPriorityHigh)
            n->streams = &streamsHi;
        else
            n->streams = &streamsLo;

        // Set dependencies
        for (auto dep : tasks[n->id].before)
        {
            Node* depPtr = getNode(dep);

            if (depPtr == nullptr)
            {
                error("Could not resolve dependency %s  -->  id %d, trying to move on", tasks[n->id].label.c_str(), dep);
                continue;
            }

            n->to.push_back(depPtr);
            depPtr->from_backup.push_back(n.get());
        }

        for (auto dep : tasks[n->id].after)
        {
            Node* depPtr = getNode(dep);

            if (depPtr == nullptr)
            {
                error("Could not resolve dependency id %d  -->  %s, trying to move on", dep, tasks[n->id].label.c_str());
                continue;
            }

            n->from_backup.push_back(depPtr);
            depPtr->to.push_back(n.get());
        }
    }
}


void TaskScheduler::removeEmptyNodes()
{
    for (auto it = nodes.begin(); it != nodes.end(); )
    {
        auto checkedNode = it->get();

        if ( tasks[checkedNode->id].funcs.size() == 0 )
        {
            debug("Task '%s' is empty and will be removed from execution", tasks[checkedNode->id].label.c_str());
            for (auto& n : nodes)
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

            it = nodes.erase(it);
        }
        else
            it++;
    }

    // Cleanup dependencies
    for (auto& n : nodes)
    {
        n->from_backup.sort();
        n->from_backup.unique();

        n->to.sort();
        n->to.unique();
    }
}

void TaskScheduler::logDepsGraph()
{
    debug("Task graph consists of total %d tasks:", nodes.size());

    for (auto& n : nodes)
    {
        std::stringstream str;

        auto& task = tasks[n->id];
        str << "Task '" << task.label << "', id " << n->id << " with " << task.funcs.size() << " functions" << std::endl;

        if (n->to.size() > 0)
        {
            str << "    Before tasks:" << std::endl;
            for (auto dep : n->to)
                str << "     * " << tasks[dep->id].label << std::endl;
        }

        if (n->from_backup.size() > 0)
        {
            str << "    After tasks:" << std::endl;
            for (auto dep : n->from_backup)
                str << "     * " << tasks[dep->id].label << std::endl;
        }

        debug("%s", str.str().c_str());
    }
}

void TaskScheduler::compile()
{
    createNodes();
    removeEmptyNodes();

    logDepsGraph();
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

    for (auto& n : nodes)
    {
        n->from = n->from_backup;

        if (n->from.empty())
            S.push(n.get());
    }

    int completed = 0;
    const int total = nodes.size();

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

                    debug("Completed group %s ", tasks[node->id].label.c_str());

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
                    error("Group '%s' raised an error",  tasks[streamNode_it->second->id].label.c_str());
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

        debug("Executing group %s on stream %lld with priority %d", tasks[node->id].label.c_str(), (int64_t)stream, node->priority);
        workMap.push_back({stream, node});

        {
            auto& task = tasks[node->id];
            NvtxCreateRange(range, task.label.c_str());
            
            for (auto& func_every : task.funcs)
                if (nExecutions % func_every.second == 0)
                    func_every.first(stream);
        }
    }

    nExecutions++;
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

void TaskScheduler::saveDependencyGraph_GraphML(std::string fname) const
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
    for (const auto& t : tasks)
        add_node(graph, t.id, t.label);

    // Edges
    for (const auto& n : nodes) {
        for (auto dep : n->to)
            add_edge(graph, n->id, dep->id);

        for (auto dep : n->from_backup)
            add_edge(graph, dep->id, n->id);
    }

    auto filename = fname + ".graphml";    
    doc.save_file(filename.c_str());
}

TaskScheduler::Task::Task(const std::string& label, TaskID id, int priority) :
    label(label),
    id(id),
    priority(priority)
{}

TaskScheduler::Node::Node(TaskID id, int priority) :
    id(id),
    priority(priority)
{}
