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

		int priority;
		std::queue<cudaStream_t>* streams;
	};

	std::vector<Node*> nodes;

	// Ordered sets of parallel work
	std::queue<cudaStream_t> streamsLo, streamsHi;

	int cudaPriorityLow, cudaPriorityHigh;

public:
	TaskScheduler();

	void addTask(std::string label, std::function<void(cudaStream_t)> task);
	void addTask(std::string label, std::vector<std::function<void(cudaStream_t)>> tasks);
	void addDependency(std::string label, std::vector<std::string> before, std::vector<std::string> after);
	void setHighPriority(std::string label);

	void compile();
	void run();

	void forceExec(std::string label);
};
