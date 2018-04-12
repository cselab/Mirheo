#include <queue>
#include <unistd.h>

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

	id = label2taskId[label] = freeTaskId++;

	auto node = std::make_unique<Node>();
	node->label = label;
	node->priority = cudaPriorityLow;
	taskId2node[id] = node.get();

	nodes.push_back(std::move(node));

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

TaskScheduler::Node* TaskScheduler::getTask(TaskID id)
{
	auto it = taskId2node.find(id);
	if (it != taskId2node.end())
		return it->second;
	else
		return nullptr;
}

TaskScheduler::Node* TaskScheduler::getTaskOrDie(TaskID id)
{
	auto node = getTask(id);
	if (node == nullptr)
		die("No such task with id %d", id);

	return node;
}




void TaskScheduler::addTask(TaskID id, std::function<void(cudaStream_t)> task, int every)
{
	Node* node = getTaskOrDie(id);

	if (every <= 0)
		die("What the fuck is this value %d???", every);

	node->funcs.push_back({task, every});
}


void TaskScheduler::addDependency(TaskID id, std::vector<TaskID> before, std::vector<TaskID> after)
{
	Node* node = getTask(id);
	if (node == nullptr)
	{
		warn("Skipping dependencies for non-existent task %d", id);
		return;
	}

	node->before.insert(node->before.end(), before.begin(), before.end());
	node->after .insert(node->after .end(), after .begin(), after .end());
}

void TaskScheduler::setHighPriority(TaskID id)
{
	Node* node = getTaskOrDie(id);

	node->priority = cudaPriorityHigh;
}

void TaskScheduler::forceExec(TaskID id)
{
	Node* node = getTask(id);

	debug("Forced execution of group %s", node->label.c_str());

	for (auto& func_every : node->funcs)
		func_every.first(0);
}

void TaskScheduler::compile()
{
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
		for (auto dep : n->before)
		{
			Node* depPtr = getTask(dep);

			if (depPtr == nullptr)
			{
				warn("Could not resolve dependency %s  -->  id %d", n->label.c_str(), dep);
				continue;
			}

			n->to.push_back(depPtr);
			depPtr->from_backup.push_back(n.get());
		}

		for (auto dep : n->after)
		{
			Node* depPtr = getTask(dep);

			if (depPtr == nullptr)
			{
				warn("Could not resolve dependency id %d  -->  %s", dep, n->label.c_str());
				continue;
			}

			n->from_backup.push_back(depPtr);
			depPtr->to.push_back(n.get());
		}
	}
}


//void TaskScheduler::fillNonEmptyNodes()
//{
//	for (auto it = nodes.begin(); it != nodes.end(); )
//	{
//		auto nPtr = it->get();
//		auto id = label2taskId[nPtr->label];
//
//		if ( nPtr->funcs.size() == 0 )
//		{
//			warn("Task '%s' is empty and will be removed from execution");
//			for (auto& n : nodes)
//			{
//				n->before.erase( std::remove(n->before.begin(), n->before.end(), id), n->before.end() );
//				n->after .erase( std::remove(n->after .begin(), n->after .end(), id), n->after .end() );
//
//				n->before.insert( n->before.end(), nPtr->before.begin(), nPtr->before.end() );
//				n->after .insert( n->after .end(), nPtr->after .begin(), nPtr->after .end() );
//			}
//		}
//	}
//}

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

					debug("Completed group %s ", node->label.c_str());

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
					error("Group '%s' raised an error",  streamNode_it->second->label.c_str());
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

		debug("Executing group %s on stream %lld with priority %d", node->label.c_str(), (int64_t)stream, node->priority);
		workMap.push_back({stream, node});

		for (auto& func_every : node->funcs)
			if (nExecutions % func_every.second == 0)
				func_every.first(stream);
	}

	nExecutions++;
	CUDA_Check( cudaDeviceSynchronize() );
}





