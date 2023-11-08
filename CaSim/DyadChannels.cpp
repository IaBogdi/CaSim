#include "DyadChannels.h"
#include "PassiveIonChannel.h"
#include "StepCurrentIonChannel.h"

#include <fstream>
#include <string>
#include <iostream>

DyadChannels::DyadChannels(nlohmann::json& j, nlohmann::json& j_jsr, int nthreads) {
	_Initialize(j, nthreads);
}

DyadChannels::DyadChannels(nlohmann::json& j, int nthreads) {
	_Initialize(j, nthreads);
}

void DyadChannels::_Initialize(nlohmann::json& j, int nthreads) {
	if (j.contains("Seed")) {
		std::vector<unsigned long int> seed = j["Seed"];
		if (seed.size() < 6)
			seed.resize(6, 0);
		rng = std::make_unique<ParallelRng>(seed.data());
	}
	else
		rng = std::make_unique<ParallelRng>();
	n_threads = nthreads;
	MCIntegral.resize(nthreads);
	SRChannels.resize(nthreads);
	SLChannels.resize(nthreads);
	for (auto const& el : j["Channels"].items()) {
			std::string s1, s2;
			if (el.value().contains("Kopen filename"))
				s1 = el.value()["Kopen filename"];
			if (el.value().contains("Kclose filename"))
				s2 = el.value()["Kclose filename"];
			auto pars_channel = el.value()["Parameters"].get<std::map<std::string, double>>();
			std::string filename(el.value()["Coordinates"]);
			std::ifstream in(filename.c_str());
			std::vector<double> g;
			for (auto const& el : el.value()["Conductance"].items()) {
				g.push_back(el.value());
			}
			double X, Y;
			std::vector<double> coords(3);
			while (in >> X >> Y) {
				coords[0] = X, coords[1] = Y;
				coords[2] = el.value()["SLChannel"] ? double(j["z"]) : 0;
				for (int i = 0; i < nthreads; ++i) {
					if (el.value()["Model"].get<std::string>().compare("StepResponse") == 0) {
						SRChannels[i].push_back(std::make_unique<StepCurrentIonChannel>(coords, el.value()["Parameters"]["Time"], g));
						is_step = true;
					}
					else {
						SRChannels[i].push_back(std::make_unique<PassiveIonChannel>(el.value()["Model"].get<std::string>(),
							coords, pars_channel, g, s1, s2));
						is_step = false;
					}
				}
			}
	}
}

int DyadChannels::GetNSRIonChannels() {
	return SRChannels[0].size();
}

int DyadChannels::GetNSLIonChannels() {
	return SLChannels[0].size();
}

std::shared_ptr <IonChannel> DyadChannels::GetIonChannel(std::vector < std::vector < std::shared_ptr < IonChannel > > >& C, int thread, int idx) {
	return C[thread][idx];
}

std::shared_ptr <IonChannel> DyadChannels::GetSRIonChannel(int thread, int idx) {
	return GetIonChannel(SRChannels,thread,idx);
}

std::shared_ptr <IonChannel> DyadChannels::GetSLIonChannel(int thread, int idx) {
	return GetIonChannel(SLChannels, thread, idx);
}

void DyadChannels::InitOpening(int thread, int channel) {
	if (is_step) {
		std::vector<double> ions;
		for (auto& channel : SRChannels[thread])
			channel->GetKineticModel()->SetMacroState(1, rng->runif(), ions_temp[thread]);
	}
	else {
		std::vector<double> ions;
		for (auto& channel : SRChannels[thread])
			channel->GetKineticModel()->SetMacroState(0, rng->runif(), ions_temp[thread]);
		SRChannels[thread][channel]->GetKineticModel()->SetMacroState(1, rng->runif(), ions_temp[thread]);
	}
}

void DyadChannels::RunMC(double dt, int n_thread, const std::vector<double>& dyad_ions, const std::vector<double>& jsr_ions, double voltage) {
	if (is_step) {
		ions_temp[n_thread]["dt"] = dt;
		if (SRChannels[n_thread][0]->GetKineticModel()->GetRate(ions_temp[n_thread]) <= 0) {
			int sr_size = SRChannels[0].size();
			for (int i = 0; i < sr_size; ++i) {
				SRChannels[n_thread][i]->GetKineticModel()->SetMacroState(0, 0, ions_temp[n_thread]);
			}
		}
	}
	else {
		int sr_size = SRChannels[0].size();
		int sl_size = SLChannels[0].size();
		int n_channels = sr_size + sl_size;
		int idx_tot = 0;
		for (int i = 0; i < sr_size; ++i) {
			for (int k = 0; k < ions_and_buffers_keys.size(); ++k)
				ions_temp[n_thread][ions_and_buffers_keys[k]] = dyad_ions[i + n_channels * (n_thread + k * n_threads)];
			ions_temp[n_thread]["Voltage"] = voltage;
			for (int k = 0; k < ions_and_buffers_keys_sr.size(); ++k)
				ions_temp[n_thread][ions_and_buffers_keys_sr[k]] = jsr_ions[i + sr_size * (n_thread + k * n_threads)];
			if (rng->runif() < SRChannels[n_thread][i]->GetKineticModel()->GetRate(ions_temp[n_thread]) * dt)
				SRChannels[n_thread][i]->GetKineticModel()->MakeTransition(rng->runif());
			for (int i = 0; i < sl_size; ++i) {
				for (int k = 0; k < ions_and_buffers_keys.size(); ++k)
					ions_temp[n_thread][ions_and_buffers_keys[k]] = dyad_ions[i + n_channels * (n_thread + k * n_threads)];
				for (int k = 0; k < ions_and_buffers_keys_sr.size(); ++k)
					ions_temp[n_thread][ions_and_buffers_keys_sr[k]] = jsr_ions[i + sr_size * (n_thread + k * n_threads)];
				if (rng->runif() < SLChannels[n_thread][i]->GetKineticModel()->GetRate(ions_temp[n_thread]) * dt)
					SLChannels[n_thread][i]->GetKineticModel()->MakeTransition(rng->runif());
			}
		}
	}
}
