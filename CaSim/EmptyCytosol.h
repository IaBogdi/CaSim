#pragma once
#include "CytosolSeries.h"
class EmptyCytosol : public CytosolSeries
{
private:
	std::vector<uint64_t> zero_dim;
	std::map<std::string, std::vector<double> > empty_conc;
	std::vector<double> extracell_ions;
public:
	EmptyCytosol(nlohmann::json& j, nlohmann::json& j_dyad) {
		for (auto const& el : j_dyad["Ions"].items()) {
			auto pars_ion = el.value();
			extracell_ions.push_back(pars_ion["Extracell"]);
		}
		for (auto const& el : j["Ions"].items()) {
			auto pars_ion = el.value();
			extracell_ions.push_back(pars_ion["Extracell"]);
		}
	}
	bool UsesGPU() {
		return 1;
	}
	void Reset() {
		return;
	}
	void GetIonsandBuffers(double*& a , double*& b) {
		return;
	}
	std::vector<double>& GetExtraCellularIons() {
		return extracell_ions;
	}

	void RunRD(double b, int a) {
		return;
	}
	void Update(double*& a , const std::vector<double>& b) {
		return;
	}
	std::map < std::string, std::vector<double> > GetConcentrations(std::vector<std::string>&) {
		return empty_conc;
	};

	std::vector<uint64_t> GetDimensions() {
		return zero_dim;
	}
};

