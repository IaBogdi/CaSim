#include "DyadChannels.h"
#include "PassiveIonChannel.h"
#include "StepCurrentIonChannel.h"

#include <fstream>
#include <string>
#include <iostream>

DyadChannels::DyadChannels(nlohmann::json& j, nlohmann::json& j_jsr, int nthreads, std::vector<float> &init_buffs) {
	Cajsrb = j_jsr["Ions"]["Calcium"]["Concentration"];
	CaM0 = init_buffs[1];
	_Initialize(j, nthreads);
}

DyadChannels::DyadChannels(nlohmann::json& j, nlohmann::json& j_jsr, int nthreads) {
	Cajsrb = j_jsr["Ions"]["Calcium"]["Concentration"];
	_Initialize(j, nthreads);
}

DyadChannels::DyadChannels(nlohmann::json& j, int nthreads) {
	_Initialize(j, nthreads);
}

void DyadChannels::_Initialize(nlohmann::json& j, int nthreads) {
	Cab = j["Ions"]["Calcium"]["Concentration"];
	Mg0 = j["Ions"]["Magnesium"]["Concentration"];
	rng = std::make_unique<ParallelRng>();
	//rng->WriteSeed("SEED.txt");
	n_threads = nthreads;
	n_ions = 2;
	n_ions_lum = 1;
	ions_temp.resize(nthreads,std::vector<double>(n_ions + n_ions_lum + 1)); //CaM
	MCIntegral.resize(nthreads);
	SRChannels.resize(nthreads);
	SLChannels.resize(nthreads);
	for (auto const& el : j["Channels"].items())
		if (!el.key().compare("RyR")) {
			std::string s1, s2;
			if (j["Channels"]["RyR"].contains("Kopen filename"))
				s1 = j["Channels"]["RyR"]["Kopen filename"];
			if (j["Channels"]["RyR"].contains("Kclose filename"))
				s2 = j["Channels"]["RyR"]["Kclose filename"];
			auto pars_channel = j["Channels"]["RyR"]["Parameters"].get<std::map<std::string, double>>();
			std::string filename(j["Channels"]["RyR"]["Coordinates"]);
			std::ifstream in(filename.c_str());
			std::vector<double> g;
			for (auto const& el : j["Channels"]["RyR"]["Conductance"].items()) {
				g.push_back(el.value());
			}
			double X, Y;
			std::vector<double> coords(3);
			while (in >> X >> Y) {
				coords[0] = X, coords[1] = Y, coords[2] = 0;
				for (int i = 0; i < nthreads; ++i) {
					if (j["Channels"]["RyR"]["Model"].get<std::string>().compare("StepResponse") == 0) {
						SRChannels[i].push_back(std::make_unique<StepCurrentIonChannel>(coords, j["Channels"]["RyR"]["Parameters"]["Time"], g));
						is_step = true;
					}
					else {
						SRChannels[i].push_back(std::make_unique<PassiveIonChannel>(j["Channels"]["RyR"]["Model"].get<std::string>(),
							coords, pars_channel, g, s1, s2));
						is_step = false;
					}
					MCIntegral[i].push_back(std::log(rng->runif()));
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
			channel->GetKineticModel()->SetMacroState(1, rng->runif(), ions);
	}
	else {
		std::vector<double> ions;
		ions.push_back(Cab);
		ions.push_back(Mg0);
		ions.push_back(CaM0);
		ions.push_back(Cajsrb);
		for (auto& channel : SRChannels[thread])
			channel->GetKineticModel()->SetMacroState(0, rng->runif(), ions);
		SRChannels[thread][channel]->GetKineticModel()->SetMacroState(1, rng->runif(), ions);
	}
}

void DyadChannels::RunMC(double dt, int n_thread, std::vector<double>& dyad_ions) {
	if (is_step) {
		std::vector<double> temp;
		temp.push_back(dt);
		MCIntegral[n_thread][0] = SRChannels[n_thread][0]->GetKineticModel()->GetRate(temp);
		if (MCIntegral[n_thread][0] <= 0) {
			int sr_size = SRChannels[0].size();
			for (int i = 0; i < sr_size; ++i) {
				SRChannels[n_thread][i]->GetKineticModel()->SetMacroState(0,0,temp);
			}
		}
	}
	else {
		int sr_size = SRChannels[0].size();
		int sl_size = SLChannels[0].size();
		int n_channels = sr_size + sl_size;
		for (int i = 0; i < sr_size; ++i) {
			for (int k = 0; k < n_ions; ++k)
				ions_temp[n_thread][k] = dyad_ions[i + n_channels * (n_thread + k * n_threads)];
			MCIntegral[n_thread][i] += SRChannels[n_thread][i]->GetKineticModel()->GetRate(ions_temp[n_thread]) * dt;
			if (MCIntegral[n_thread][i] > 0) {
				SRChannels[n_thread][i]->GetKineticModel()->MakeTransition(rng->runif());
				MCIntegral[n_thread][i] = std::log(rng->runif());
			}
		}
		for (int i = 0; i < sl_size; ++i) {
			for (int k = 0; k < n_ions; ++k)
				ions_temp[n_thread][k] = dyad_ions[sr_size + i + n_channels * (n_thread + k * n_threads)];
			MCIntegral[n_thread][sr_size + i] += SLChannels[n_thread][i]->GetKineticModel()->GetRate(ions_temp[n_thread]) * dt;
			if (MCIntegral[n_thread][sr_size + i] > 0) {
				SLChannels[n_thread][i]->GetKineticModel()->MakeTransition(rng->runif());
				MCIntegral[n_thread][sr_size + i] = std::log(rng->runif());
			}
		}
	}
}

void DyadChannels::RunMC(double dt, int n_thread, std::vector<double>& dyad_ions, const std::vector<double>& jsr_ions) {
	if (is_step) {
		std::vector<double> temp;
		temp.push_back(dt);
		MCIntegral[n_thread][0] = SRChannels[n_thread][0]->GetKineticModel()->GetRate(temp);
		if (MCIntegral[n_thread][0] <= 0) {
			int sr_size = SRChannels[0].size();
			for (int i = 0; i < sr_size; ++i) {
				SRChannels[n_thread][i]->GetKineticModel()->SetMacroState(0, 0, temp);
			}
		}
	}
	else {
		int sr_size = SRChannels[0].size();
		int sl_size = SLChannels[0].size();
		int n_channels = sr_size + sl_size;
		for (int i = 0; i < sr_size; ++i) {
			for (int k = 0; k < n_ions; ++k)
				ions_temp[n_thread][k] = dyad_ions[i + n_channels * (n_thread + k * n_threads)];
			int jj = 1;
			ions_temp[n_thread][n_ions] = dyad_ions[i + n_channels * (n_thread + (jj + n_ions) * n_threads)];
			for (int k = 0; k < n_ions_lum; ++k)
				ions_temp[n_thread][k + n_ions + 1] = jsr_ions[i + sr_size * (n_thread + k * n_threads)];
			if (rng->runif() < SRChannels[n_thread][i]->GetKineticModel()->GetRate(ions_temp[n_thread]) * dt)
				SRChannels[n_thread][i]->GetKineticModel()->MakeTransition(rng->runif());
			/*
			MCIntegral[n_thread][i] += SRChannels[n_thread][i]->GetKineticModel()->GetRate(ions_temp[n_thread]) * dt;
			if (MCIntegral[n_thread][i] > 0) {
				auto kineticModel = SRChannels[n_thread][i]->GetKineticModel();
				double dt2;
				do {
					dt2 = MCIntegral[n_thread][i] / kineticModel->GetRate(ions_temp[n_thread]);
					MCIntegral[n_thread][i] = std::log(rng->runif());
					kineticModel->MakeTransition(rng->runif());
					MCIntegral[n_thread][i] += kineticModel->GetRate(ions_temp[n_thread]) * dt2;
				} while (MCIntegral[n_thread][i] > 0);
			}
			*/
			for (int i = 0; i < sl_size; ++i) {
				for (int k = 0; k < n_ions; ++k)
					ions_temp[n_thread][k] = dyad_ions[sr_size + i + n_channels * (n_thread + k * n_threads)];
				MCIntegral[n_thread][sr_size + i] += SLChannels[n_thread][i]->GetKineticModel()->GetRate(ions_temp[n_thread]) * dt;
				if (MCIntegral[n_thread][sr_size + i] > 0) {
					SLChannels[n_thread][i]->GetKineticModel()->MakeTransition(rng->runif());
					MCIntegral[n_thread][sr_size + i] = std::log(rng->runif());
				}
			}
		}
	}
}