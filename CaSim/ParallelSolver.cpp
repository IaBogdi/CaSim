#include "ParallelSolver.h"

using json = nlohmann::json;

ParallelSolver::ParallelSolver(json& file) {
	n_threads = file["NumThreads"] <= 0 ? omp_get_max_threads() : int(file["NumThreads"]);
	write_data = file["WriteData"];
	experiment_name = std::string(file["Name"]);
	fs::create_directory(experiment_name);
	if (write_data > 0)
		fs::create_directory(experiment_name + std::string("/results"));
	fs::create_directory(experiment_name + std::string("/analysis"));
	std::string write_pars = experiment_name + std::string("/Parameters.json");
	std::ofstream o(write_pars.c_str());
	o << std::setw(4) << file << std::endl;
	o.close();
	if (file["Calcium Release Site"]["Type"] == "QuasyStationary") {
		for (int i = 0; i < n_threads; ++i)
			crs_vec.push_back(std::make_shared<QuasyStationaryReleaseSite>(file));
		
	}
	rng = std::make_unique<ParallelRng>();
	time_max = file["Calcium Release Site"]["Max Time"];
	nadir = file["Nadir"];
}

void ParallelSolver::WriteFiles(int n_threads) {
	std::string names[4] = { std::string("Channels_"),std::string("Concentrations_Dyadic_Cleft_"),std::string("Concentrations_Cytosol_"), std::string("Concentrations_Lumen_") };
#pragma omp parallel for schedule(dynamic) num_threads(n_threads)
	for (int j = 0; j < 4; ++j) {
		std::ofstream out;
		std::vector<std::string> variables;
		for (auto const& imap : out_vars[j])
			variables.push_back(imap.first);
		for (auto const& var : variables) {
			std::string filename = experiment_name + std::string("/results/") + names[j] + var + std::string(".txt");
			out.open(filename.c_str());
			for (const auto& scenario : out_vars[j][var]) {
				for (const auto& snapshot : scenario) {
					for (const auto& num : snapshot)
						out << num << " ";
					out << std::endl;
				}
				out << " " << std::endl;
			}
			out.close();
		}
	}
}

void ParallelSolver::RunSimulations(long long n_scenarios) {
	long long sc_per_thread = n_scenarios / n_threads;
	int percent = 0;
	//run 1st scenario in order to initialize out_vars
	datavector channels_t; // channel, time, state
	datavector concentrations_dc_t; // buffer, time, data(pos/conc)
	datavector concentrations_lumen_t; // buffer, time, data(pos/conc)
	datavector concentrations_cyt_t; // buffer, time, data(pos/conc)
	std::string names[4] = { std::string("Channels_"),std::string("Concentrations_Dyadic_Cleft_"),std::string("Concentrations_Cytosol_"), std::string("Concentrations_Lumen_") };
	crs_vec[0]->RunSimulation(channels_t, concentrations_dc_t, concentrations_lumen_t, concentrations_cyt_t, 0, rng.get());
	datavector* datas[4] = { &channels_t, &concentrations_dc_t, &concentrations_cyt_t, &concentrations_lumen_t };
	for (int j = 0; j < 4; ++j) {
		std::vector<std::string> variables;
		for (auto const& imap : *datas[j])
			variables.push_back(imap.first);
		for (auto const& var : variables) {
			std::vector < std::vector < double > > v2;
			std::vector < std::vector < std::vector < double > > >  v(n_scenarios, v2);
			out_vars[j][var] = v;
			out_vars[j][var][0] = datas[j]->at(var);
		}
	}
	int sc_per_channel = n_scenarios / crs_vec[0]->GetNChannels();

	#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
	for (long long i = 1; i < n_scenarios; ++i) {
		int n_thr = omp_get_thread_num();
		if (i *100 / n_scenarios > percent) {
			++percent;
			std::cout << percent << "% completed" << std::endl;
		}
		datavector channels; // channel, time, state
		datavector concentrations_dc; // buffer, time, data(pos/conc)
		datavector concentrations_lumen; // buffer, time, data(pos/conc)
		datavector concentrations_cyt; // buffer, time, data(pos/conc)
		std::string names[4] = {std::string("Channels_"),std::string("Concentrations_Dyadic_Cleft_"),std::string("Concentrations_Cytosol_"), std::string("Concentrations_Lumen_")};
		crs_vec[n_thr]->RunSimulation(channels,concentrations_dc,concentrations_lumen,concentrations_cyt, i / sc_per_channel, rng.get());
		datavector *datas[4] = { &channels, &concentrations_dc, &concentrations_cyt, &concentrations_lumen };
		for (int j = 0; j < 4; ++j) {
			std::vector<std::string> variables;
			for (auto const& imap : *datas[j])
				variables.push_back(imap.first);
			for (auto const& var : variables)
				out_vars[j][var][i] = datas[j]->at(var);
		}
	}
	std::cout << "Computation is finished" << std::endl;
	//write raw data and primary analysis in parallel
	int threads_part_two = n_threads >= 2 ? 2 : 1;
	int threads_writing_part = -1;
	if (write_data > 0) {
		//check number of possible categories
		for (int j = 0; j < 4; ++j) {
			std::vector<std::string> variables;
			for (auto const& imap : out_vars[j])
				variables.push_back(imap.first);
			if (variables.size() > 0)
				++threads_writing_part;
		}
		threads_writing_part = n_threads - 1 - threads_writing_part >= 1 ? threads_writing_part : 1;
		omp_set_nested(1);
	}
	else {
		omp_set_nested(0);
	}
	int threads_part_three;
	#pragma omp parallel    
	{
	#pragma omp sections
		{
	#pragma omp section
			{
				if (write_data > 0) {
					std::cout << "Writing results" << std::endl;
					WriteFiles(threads_writing_part);
				}
			}
	#pragma omp section
			{
				std::cout << "Preparing preliminary analysis" << std::endl;
				threads_part_three = n_threads - 1 - threads_writing_part >= 1 ? n_threads - threads_writing_part : 1;
				std::vector<std::string> variables;
				for (auto const& imap : out_vars[0])
					variables.push_back(imap.first);
				//since it's only one channel, we can do it like that
				auto channel_map = crs_vec[0]->GetChannelsMacroStatesMap(variables[0]);
				analyzer = std::make_unique<IndChannelCREAnalyzer>();
				analyzer->RunAnalysis(experiment_name,out_vars[0][variables[0]], channel_map,time_max,threads_part_three,nadir);
			}
		}
	}
	std::cout << "Done" << std::endl;
}
