#include "../core/iniparser.h"
#include "timer.h"

Logger logger;


int main()
{
	IniParser parser("../config.cfg");

	std::cout << "============================================================" << std::endl;

	std::cout << parser.getInt("Section1", "g", -100) << std::endl;
	std::cout << parser.getDouble("Section2", "num", -1.234) << std::endl;
	std::cout << parser.getDouble("Section3", "num", 144) << std::endl;
	std::cout << parser.getString("Section1", "ddd") << std::endl;
	std::cout << parser.getString("Section1", "b") << std::endl;
	std::cout << parser.getInt("Section2", "nn", -1) << std::endl;

	std::cout << "============================================================" << std::endl;


	double res = 0;
	const int iters = 1000;
	Timer timer;
	timer.start();
	for (int i=0; i<iters; i++)
	{
		int a = parser.getInt("Section1", "g", -100);
		double b = parser.getDouble("Section2", "num", -1.234);
		double c = 0;//parser.getDouble("Section3", "num", 144);
		//parser.getString("Section1", "ddd") << std::endl;
		int d = parser.getInt("Section2", "nn", -1);

		res += a+b+c+d;
	}

	std::cout << "1 step is " << timer.elapsed() / (1000.0 * iters) << " us    " << res << std::endl;

	return 0;
}
