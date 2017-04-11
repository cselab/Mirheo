#include <queue>

#include <core/task_scheduler.h>
#include <core/logger.h>

void TaskScheduler::addTask(std::string label, std::function<void()> task)
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

void TaskScheduler::addDependency(std::string label, std::vector<std::string> dependsOn)
{
	Node* node = nullptr;
	for (auto n : nodes)
		if (n->label == label) node = n;

	if (node == nullptr)
		die("Task group with label %s not found", label.c_str());

	for (auto& id : dependsOn)
	{
		Node* dep = nullptr;
		for (auto n : nodes)
			if (n->label == id) dep = n;

		if (node == nullptr)
			die("Task group with label %s not found", id.c_str());

		edges.push_back({dep, node});
	}
}

inline bool TaskScheduler::isIndependent(Node* n)
{
	bool isolated = true;
	for (auto& e : edges)
		if (e.to == n)
		{
			isolated = false;
			break;
		}

	return isolated;
}

void TaskScheduler::compile()
{
	// Kahn's algorithm
	// https://en.wikipedia.org/wiki/Topological_sorting

	std::queue<Node*> S;

	for (auto n : nodes)
		if (isIndependent(n))
			S.push(n);

	while (S.size() > 0)
	{
		Node* node = S.front();
		S.pop();
		sorted.push_back(node);

		auto it = std::begin(edges);
		while (it != std::end(edges))
		{
		    if (it->from == node)
		    {
		        Node* m = it->to;
		        it = edges.erase(it);

		        if (isIndependent(m))
		        	S.push(m);
		    }
		    else
		    {
		        it++;
		    }
		}
	}

	if (edges.size())
		die("Wrong task graph: probably loop exists");
}

void TaskScheduler::run()
{
	for (auto node : sorted)
	{
		debug("Executing group %s", node->label.c_str());
		for (auto func : node->funcs)
			func();
	}
}





