// Copyright 2020 ETH Zurich. All Rights Reserved.
#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace mirheo
{

/** \brief CUDA-aware task scheduler

    Manages task dependencies and run them concurrently on different CUDA streams.
    This is designed to be run in a time stepping scheme, e.g. all the tasks of a
    single time step must be described here before calling the run() method repetitively.
 */
class TaskScheduler
{
public:
    /// Represents the unique id of a task
    using TaskID = int;
    /// Represents the function performed by a task. Will be executed on the given stream.
    using Function = std::function<void(cudaStream_t)>;

    /// Special task id value to represent invalid tasks
    static constexpr TaskID invalidTaskId {static_cast<TaskID>(-1)};

    /// Default constructor
    TaskScheduler();
    ~TaskScheduler();

    /** \brief Create and register an empty task named \p label
        \param [in] label The name of the task
        \return the task id associated with the new task

        This method will die if a task with the given label already exists.
    */
    TaskID createTask(const std::string& label);

    /** \brief Retrieve the task id of the task with a given label
        \param [in] label The name of the task
        \return the task id if it exists, or \c invalidTaskId if it doesn't
    */
    TaskID getTaskId(const std::string& label);

    /** \brief Retrieve the task id of the task with a given label
        \param [in] label The name of the task
        \return the task id

        This method will die if no registered task has the given label
    */
    TaskID getTaskIdOrDie(const std::string& label);

    /** \brief Add a function to execute to the given task.
        \param [in] id Task Id
        \param [in] task The function to execute
        \param [in] execEvery Execute his function every this number of calls of run().

        Multiple functions can be added in a single task.
        The order of execution of these functions is the order in which they were added.
        This method will fail if the required task does not exist.
     */
    void addTask(TaskID id, Function task, int execEvery = 1);

    /** \brief add dependencies around a given task
        \param [in] id The task that must be executed before \p before and after \p after
        \param [in] before the list of tasks that must be executed after the task with id \p id
        \param [in] after the list of tasks that must be executed before the task with id \p id
     */
    void addDependency(TaskID id, std::vector<TaskID> before, std::vector<TaskID> after);

    /** \brief Set the execution of a task to high priority.
        \param [in] id The task id
     */
    void setHighPriority(TaskID id);

    /** \brief Prepare the internal state so that the scheduler can perform execution of all tasks.
        No other calls related to task creation / modification / dependencies must be performed after
        calling this function.
     */
    void compile();

    /** Execute the tasks in the order required by the given dependencies and priorities.
        Must be called after compile().
     */
    void run();

    /** Dump a representation of the tasks and their dependencies in graphML format.
        \param [in] fname The file name to dump the graph to (without extension).
     */
    void dumpGraphToGraphML(const std::string& fname) const;

    /** Execute a given task on a given stream
        \param [in] id the task to execute
        \param [in] stream The stream to execute the task
     */
    void forceExec(TaskID id, cudaStream_t stream);

private:

    struct Task
    {
        Task(const std::string& label, TaskID id, int priority);

        std::string label;
        TaskID id;
        int priority;

        std::vector< std::pair<Function, int> > funcs;
        std::vector<TaskID> before, after;
    };

    struct Node;
    struct Node
    {
        Node(TaskID id, int priority);
        TaskID id;

        std::list<Node*> to, from, from_backup;

        int priority;
        std::queue<cudaStream_t>* streams;
    };

    std::vector<Task> tasks_;
    std::vector< std::unique_ptr<Node> > nodes_;

    // Ordered sets of parallel work
    std::queue<cudaStream_t> streamsLo_, streamsHi_;

    int cudaPriorityLow_, cudaPriorityHigh_;

    int nExecutions_{0};

    std::unordered_map<std::string, TaskID> label2taskId_;

    void _checkTaskExistsOrDie(TaskID id) const;
    Node* _getNode     (TaskID id);
    Node* _getNodeOrDie(TaskID id);

    void _createNodes();
    void _removeEmptyNodes();
    void _logDepsGraph();

};

} // namespace mirheo
