/*
 * task_scheduler.h
 *
 *  Created on: Apr 10, 2017
 *      Author: alexeedm
 */

#include <string>
#include <vector>
#include <functional>
#include <list>
#include <queue>
#include <unordered_map>
#include <memory>

#include <cuda_runtime.h>

class TaskScheduler
{
public:
    using TaskID = int;
    static const TaskID invalidTaskId = (TaskID) -1;

    TaskScheduler();

    TaskID createTask     (const std::string& label);
    TaskID getTaskId      (const std::string& label);
    TaskID getTaskIdOrDie (const std::string& label);

    void addTask(TaskID id, std::function<void(cudaStream_t)> task, int execEvery = 1);
    void addDependency(TaskID id, std::vector<TaskID> before, std::vector<TaskID> after);
    void setHighPriority(TaskID id);

    void compile();
    void run();
    void saveDependencyGraph_GraphML(std::string fname) const;

    void forceExec(TaskID id, cudaStream_t stream);

private:

    struct Task
    {
        std::string label;
        TaskID id;
        int priority;

        std::vector< std::pair<std::function<void(cudaStream_t)>, int> > funcs;
        std::vector<TaskID> before, after;
    };

    struct Node;
    struct Node
    {
        TaskID id;

        std::list<Node*> to, from, from_backup;

        int priority;
        std::queue<cudaStream_t>* streams;
    };

    std::vector<Task> tasks;
    std::vector< std::unique_ptr<Node> > nodes;

    // Ordered sets of parallel work
    std::queue<cudaStream_t> streamsLo, streamsHi;

    int cudaPriorityLow, cudaPriorityHigh;

    int nExecutions{0};

    std::unordered_map<std::string, TaskID> label2taskId;

    Node* getNode     (TaskID id);
    Node* getNodeOrDie(TaskID id);

    void createNodes();
    void removeEmptyNodes();
    void logDepsGraph();

};
