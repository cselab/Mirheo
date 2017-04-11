/*
 * task_scheduler.h
 *
 *  Created on: Apr 10, 2017
 *      Author: alexeedm
 */

#include <string>
#include <vector>
#include <functional>

class TaskScheduler
{
private:
	struct Node
	{
		std::string label;
		std::vector<std::function<void()>> funcs;
	};

	struct Edge
	{
		Node *from, *to;
	};

	std::vector<Node*> nodes;
	std::vector<Edge> edges;

	std::vector<Node*> sorted;

	inline bool isIndependent(Node* n);

public:
	void addTask(std::string label, std::function<void()> task);
	void addDependency(std::string label, std::vector<std::string> dependsOn);
	void compile();
	void run();
};
