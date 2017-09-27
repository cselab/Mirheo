#include <queue>
#include <unistd.h>

#include <core/task_scheduler.h>
#include <core/logger.h>

TaskScheduler::TaskScheduler()
{
	CUDA_Check( cudaDeviceGetStreamPriorityRange(&cudaPriorityLow, &cudaPriorityHigh) );
}

void TaskScheduler::addTask(std::string label, std::function<void(cudaStream_t)> task)
{
	Node* node = nullptr;
	for (auto n : nodes)
		if (n->label == label) node = n;

	if (node == nullptr)
	{
		node = new Node();
		node->label = label;
		node->priority = cudaPriorityLow;
		nodes.push_back(node);
	}

	node->funcs.push_back(task);
}

void TaskScheduler::addTask(std::string label, std::vector<std::function<void(cudaStream_t)>> tasks)
{
	for (auto& t : tasks)
		addTask(label, t);
}

void TaskScheduler::addDependency(std::string label, std::vector<std::string> before, std::vector<std::string> after)
{
	Node* node = nullptr;
	for (auto n : nodes)
		if (n->label == label) node = n;

	if (node == nullptr)
		die("Task group with label %s not found", label.c_str());

	node->before.insert(node->before.end(), before.begin(), before.end());
	node->after .insert(node->after .end(), after .begin(), after .end());
}

void TaskScheduler::setHighPriority(std::string label)
{
	Node* node = nullptr;
	for (auto n : nodes)
		if (n->label == label) node = n;

	if (node == nullptr)
		die("Task group with label %s not found", label.c_str());

	node->priority = cudaPriorityHigh;
}

void TaskScheduler::forceExec(std::string label)
{
	Node* node = nullptr;
	for (auto n : nodes)
		if (n->label == label) node = n;

	if (node == nullptr)
		die("Task group with label %s not found", label.c_str());

	for (auto& func : node->funcs)
		func(0);
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
		for (auto& dep : n->before)
		{
			Node* depPtr = nullptr;
			for (auto ndep : nodes)
				if (ndep->label == dep)
				{
					depPtr = ndep;
					break;
				}

			if (depPtr == nullptr)
				die("Could not resolve dependency %s  -->  %s", n->label.c_str(), dep.c_str());

			n->to.push_back(depPtr);
			depPtr->from_backup.push_back(n);
		}

		for (auto& dep : n->after)
		{
			Node* depPtr = nullptr;
			for (auto ndep : nodes)
				if (ndep->label == dep)
				{
					depPtr = ndep;
					break;
				}

			if (depPtr == nullptr)
				die("Could not resolve dependency %s  -->  %s", dep.c_str(),  n->label.c_str());

			n->from_backup.push_back(depPtr);
			depPtr->to.push_back(n);
		}
	}
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

	for (auto n : nodes)
	{
		n->from = n->from_backup;

		if (n->from.empty())
			S.push(n);
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
				if ( cudaStreamQuery(streamNode_it->first) == cudaSuccess )
				{
					auto node = streamNode_it->second;

					info("Completed group %s ", node->label.c_str());

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
				else
					streamNode_it++;
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

		info("Executing group %s on stream %lld with priority %d", node->label.c_str(), (int64_t)stream, node->priority);
		workMap.push_back({stream, node});

		for (auto& func : node->funcs)
			func(stream);
	}

	CUDA_Check( cudaDeviceSynchronize() );
}





