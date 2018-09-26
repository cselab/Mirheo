#include <queue>
#include <unistd.h>
#include <sstream>
#include <fstream>

#include <core/task_scheduler.h>
#include <core/logger.h>
#include <core/utils/make_unique.h>

TaskScheduler::TaskScheduler()
{
    CUDA_Check( cudaDeviceGetStreamPriorityRange(&cudaPriorityLow, &cudaPriorityHigh) );
}

TaskScheduler::TaskID TaskScheduler::createTask(const std::string& label)
{
    auto id = getTaskId(label);
    if (id != invalidTaskId)
        die("Task '%s' already exists", label.c_str());

    id = tasks.size();
    label2taskId[label] = id;

    Task task {label, id};
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


void TaskScheduler::addTask(TaskID id, std::function<void(cudaStream_t)> task, int every)
{
    if (id >= tasks.size() || id < 0)
        die("No such task with id %d", id);

    if (every <= 0)
        die("What the fuck is this value %d???", every);

    tasks[id].funcs.push_back({task, every});
}


void TaskScheduler::addDependency(TaskID id, std::vector<TaskID> before, std::vector<TaskID> after)
{
    if (id >= tasks.size() || id < 0)
        die("No such task with id %d", id);

    tasks[id].before.insert(tasks[id].before.end(), before.begin(), before.end());
    tasks[id].after .insert(tasks[id].after .end(), after .begin(), after .end());
}

void TaskScheduler::setHighPriority(TaskID id)
{
    if (id >= tasks.size() || id < 0)
        die("No such task with id %d", id);

    tasks[id].priority = cudaPriorityHigh;
}

void TaskScheduler::forceExec(TaskID id, cudaStream_t stream)
{
    if (id >= tasks.size() || id < 0)
        die("No such task with id %d", id);

    debug("Forced execution of group %s", tasks[id].label.c_str());

    for (auto& func_every : tasks[id].funcs)
        func_every.first(stream);
}


void TaskScheduler::createNodes()
{
    nodes.clear();

    for (auto& t : tasks)
    {
        auto node = std::make_unique<Node>();

        node->id = t.id;
        node->priority = t.priority;

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
            warn("Task '%s' is empty and will be removed from execution", tasks[checkedNode->id].label.c_str());
            for (auto& n : nodes)
            {
                int toSize = n->to.size();
                int from_backupSize = n->from_backup.size();

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
    info("Task graph consists of total %d tasks:", nodes.size());

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

        info("%s", str.str().c_str());
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

    auto compareNodes = [] (Node* a, Node* b) {
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
            CUDA_Check( cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, node->priority) );
        else
        {
            stream = node->streams->front();
            node->streams->pop();
        }

        debug("Executing group %s on stream %lld with priority %d", tasks[node->id].label.c_str(), (int64_t)stream, node->priority);
        workMap.push_back({stream, node});

        for (auto& func_every : tasks[node->id].funcs)
            if (nExecutions % func_every.second == 0)
                func_every.first(stream);
    }

    nExecutions++;
    CUDA_Check( cudaDeviceSynchronize() );
}

// TODO: use pugixml
void TaskScheduler::saveDependencyGraph_GraphML(std::string fname)
{
    std::ofstream fout(fname);

    // Header
    fout << R"(<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"  
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns 
     http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">)" << "\n" << "\n";

    fout << R"(    <key id="label" for="node" attr.name="label" attr.type="string"/>)" << "\n" << "\n";

    fout << R"(    <graph id="Task graph" edgedefault="directed">)" << "\n";

    // Nodes
    for (auto& t : tasks)
    {
        fout << "        <node id=\"" << t.id << "\" >\n";
        fout << "            <data key=\"label\"> " << t.label << "</data>\n";
        fout << "        </node>\n";
    }

    // Edges
    for (auto& n : nodes)
    {

        for (auto dep : n->to)
            fout << "        <edge source=\"" << n->id << "\" target=\"" << dep->id << "\" />\n";

        for (auto dep : n->from_backup)
            fout << "        <edge source=\"" << dep->id << "\" target=\"" << n->id << "\" />\n";
    }

    // Footer
    fout << R"(    </graph>
</graphml>)" << "\n";
}



