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

class TaskScheduler
{
private:

	struct Node;
	struct Node
	{
		std::string label;
		std::vector<std::function<void(cudaStream_t)>> funcs;

		std::vector<std::string> before, after;
		std::list<Node*> to, from, from_backup;
	};

	std::vector<Node*> nodes;

	// Ordered sets of parallel work
	std::queue<cudaStream_t> streams;

public:
	void addTask(std::string label, std::function<void(cudaStream_t)> task);
	void addTask(std::string label, std::vector<std::function<void(cudaStream_t)>> tasks);
	void addDependency(std::string label, std::vector<std::string> before, std::vector<std::string> after);

	void compile();
	void run();
};
