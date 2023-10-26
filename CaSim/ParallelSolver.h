#pragma once
#include <iostream>
#include <vector>
#include "CaSolver.h"
#include "QuasyStationaryReleaseSite.h"
#include "IndChannelCREAnalyzer.h"
#include <memory>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <omp.h>
#include <filesystem>
#include <array>
#include "ParallelRng.h"
namespace fs = std::filesystem;


class ParallelSolver : public CaSolver {
	std::vector < std::shared_ptr < CalciumReleaseSite > > crs_vec;
	std::vector < std::vector < std::string > > datas_names;
	std::unique_ptr<IndChannelCREAnalyzer> analyzer;
	int n_threads;
	std::string experiment_name;
	std::unique_ptr<ParallelRng> rng;
	std::array < dataset, 4> out_vars;
	void WriteFiles(int);
	double time_max;
	int write_data; 
	int nadir;
public:
	ParallelSolver(nlohmann::json&);
	void RunSimulations(long long n_scenarios);
};

