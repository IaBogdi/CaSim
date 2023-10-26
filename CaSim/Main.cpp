#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <chrono>

#include "ParallelHybridSolver.h"
#include "ParallelSolver.h"

using json = nlohmann::json;

using namespace std;

int main(int argc, char** argv) {
	std::ifstream file;
	if (argc > 1) {
		file.open(argv[1]);
	}
	else {
		std::cout << "Write name of configuration file: " << std::endl;
		std::string fname;
		std::cin >> fname;
		file.open(fname.c_str());
	}
	json j = json::parse(file);
	string s(j["Solver"]);
	unique_ptr<CaSolver> solver;
	if (!s.compare("ParallelSolver")) {
		solver = make_unique<ParallelSolver>(j);
	}
	else if (!s.compare("ParallelHybridSolver")) {
		solver = make_unique<ParallelHybridSolver>(j);
	}
	else {
		cout << "Wrong Solver Parameter" << endl;
		return -1;
	}
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	solver->RunSimulations(j["NumScenarios"]);
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> exec_time = end - start;
	file.close();
	std::cout << "Execution time:" << exec_time.count() << " seconds" << std::endl;
	//system("pause");
	return 0;
}