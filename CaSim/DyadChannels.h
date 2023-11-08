#pragma once
#include "IonChannel.h"
#include "Structures.h"
#include "ParallelRng.h"
#include <nlohmann/json.hpp>
#include <memory>

class DyadChannels
{
	std::vector < std::vector < std::shared_ptr < IonChannel > > > SRChannels;
	std::vector < std::vector < std::shared_ptr < IonChannel > > > SLChannels;
	std::vector < std::vector <double> > MCIntegral;
	std::unique_ptr<ParallelRng> rng;
	std::shared_ptr <IonChannel> GetIonChannel(std::vector < std::vector < std::shared_ptr < IonChannel > > >&, int, int);
	int n_threads;
	int n_ions;
	int n_buf;
	bool is_step;
	std::vector<std::unordered_map<std::string, double> > ions_temp;
	std::vector<std::string> ions_and_buffers_keys;
	std::vector<std::string> ions_and_buffers_keys_sr;
	void _Initialize(nlohmann::json&, int);
public:
	DyadChannels(nlohmann::json&, nlohmann::json&, int);
	template <typename T>
	DyadChannels(nlohmann::json& j, nlohmann::json& j_jsr, int nthreads, std::vector < std::unique_ptr <Ion<T> > >& ions, std::vector < std::unique_ptr <Buffer<T> > >& buffers, double Voltage) {
		//construct map of ions and buffers so that it saves initial values
		std::unordered_map<std::string, double> ions_and_bufs;
		n_ions = 0;
		for (auto& ion : ions) {
			ions_and_bufs[ion->name] = ion->Cb;
			ions_and_buffers_keys.push_back(ion->name);
		}
		for (auto& buf : buffers)
			ions_and_bufs[buf->name] = buf->Ctot;
		for (auto& ion : ions) {
			for (auto& buf : buffers) {
				auto it = buf->ions_kinetics.find(ion->name);
				if (it != buf->ions_kinetics.end()) {
					auto type_name = ion->name + std::string("-") + buf->name;
					ions_and_bufs[type_name] = buf->ions_kinetics[ion->name]->initial_C;
					ions_and_buffers_keys.push_back(type_name);
					ions_and_bufs[buf->name] -= ions_and_bufs[type_name];
				}
			}
		}
		ions_and_bufs["Voltage"] = Voltage;
		for (auto& buf : buffers)
			ions_and_buffers_keys.push_back(buf->name);
		for (auto& ion : ions) {
			auto sr_name = std::string("SR") + ion->name;
			if (j_jsr["Ions"].contains(ion->name))
				ions_and_bufs[sr_name] = j_jsr["Ions"][ion->name]["Concentration"];
			else
				ions_and_bufs[sr_name] = 0;
			ions_and_buffers_keys_sr.push_back(sr_name);
		}
		ions_temp.resize(nthreads, ions_and_bufs);
		_Initialize(j, nthreads);
	}
	DyadChannels(nlohmann::json&, int);
	std::shared_ptr <IonChannel> GetSRIonChannel(int, int);
	std::shared_ptr <IonChannel> GetSLIonChannel(int, int);
	int GetNSRIonChannels();
	int GetNSLIonChannels();
	void InitOpening(int, int);
	void RunMC(double, int, const std::vector<double>&, const std::vector<double>&, double);
};

