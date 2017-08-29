#include <queue>
#include <unistd.h>

#include <core/task_scheduler.h>
#include <core/logger.h>

void TaskScheduler::addTask(std::string label, std::function<void(cudaStream_t)> task)
{
	Node* node = nullptr;
	for (auto n : nodes)
		if (n->label == label) node = n;

	if (node == nullptr)
	{
		node = new Node();
		node->label = label;
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

void TaskScheduler::compile()
{
	for (auto n : nodes)
	{
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

	std::queue<Node*> S;
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
			usleep(1);
			for (auto streamNode_it = workMap.begin(); streamNode_it != workMap.end(); )
			{
				if ( cudaStreamQuery(streamNode_it->first) == cudaSuccess )
				{
					streams.push(streamNode_it->first);

					auto node = streamNode_it->second;
					for (auto dep : node->to)
					{
						dep->from.remove(node);
						if (dep->from.empty())
							S.push(dep);
					}

					completed++;
					streamNode_it = workMap.erase(streamNode_it);
				}
				else
					streamNode_it++;
			}
		}

		if (completed == total)
			break;

		Node* node = S.front();
		S.pop();

		cudaStream_t stream;
		if (streams.empty())
			CUDA_Check( cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0) );
		else
		{
			stream = streams.front();
			streams.pop();
		}

		debug("Executing group %s on stream %lld", node->label.c_str(), (int64_t)stream);
		workMap.push_back({stream, node});

		for (auto& func : node->funcs)
			func(stream);
	}

	for (auto& stream_node : workMap)
		CUDA_Check( cudaStreamSynchronize(stream_node.first) );
}





