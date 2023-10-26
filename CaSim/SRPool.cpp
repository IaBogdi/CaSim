#include "SRPool.h"
#include <iostream>

SRPool::SRPool(nlohmann::json& j, int n_threads, double V_elem,int n_channels, int n_ions) {
	CQ_conc = j["Total CQ (uM)"];
	for (auto const& el : j["Ions"].items()) {
		auto pars_ion = el.value();
		nCQ_sites.push_back(pars_ion["Binding sites of CQ"]);
		K_CQ.push_back(pars_ion["K CQ (uM)"]);
		tau_refill.push_back(pars_ion["T refill (ms)"]);
		ions_0.push_back(pars_ion["Concentration"]);
		ion_names.push_back(el.key());
	}
	V = j["V jSR (nm3)"];
	V_dyad_element = V_elem;
	for (int i = 0; i < ions_0.size(); ++i) {
		for (int j = 0; j < n_channels; ++j) {
			for (int k = 0; k < n_threads; ++k) {
				ions.push_back(ions_0[i]);
				ions_new.push_back(ions_0[i]);
			}
		}
	}
	for (int i = 0; i < ions_0.size(); ++i) {
		for (int k = 0; k < n_threads; ++k) {
			ions_NSR.push_back(ions_0[i]);
			IjSR_dyad.push_back(0);
			}
	}
	nthreads = n_threads;
	nchannels = n_channels;
	nions = ions_0.size();
	sr_dims.resize(3);
	sr_dims[1] = nthreads;
	sr_dims[2] = nchannels;
	is_const = j["Constant"];
}

std::vector<double>& SRPool::GetIons() {
	return ions_new;
}

inline long SRPool::_GetIdx(int ion, int channel, int thread) {
	return channel + nchannels * (thread + ion * nthreads);
}

void SRPool::Update(const std::vector<double>& total_jsr_current, const std::vector<double>& nsr_ions) {
	if (!is_const) {
		for (int i = 0; i < nions; ++i) {
			for (int k = 0; k < nthreads; ++k) {
				IjSR_dyad[k + i * nthreads] = total_jsr_current[k + i * nthreads] * V_dyad_element;
				ions_NSR[k + i * nthreads] = nsr_ions[k + i * nthreads];
				for (int j = 0; j < nchannels; ++j) {
					int idx = _GetIdx(i, j, k);
					ions[idx] = ions_new[idx];
				}
			}
		}
	}
}

void SRPool::Run(double dt, int idx) {
	if (!is_const) {
		double Iref;
		double beta = 1;
		for (int j = 0; j < nions; ++j) {
			beta += nCQ_sites[j] * K_CQ[j] * CQ_conc / ((K_CQ[j] + ions[idx + j * nthreads]) * (K_CQ[j] + ions[idx + j * nthreads]));
		}
		int idx_dyad = 0;
		for (int i = 0; i < nions; ++i) {
			idx_dyad = _GetIdx(i, 0, idx);
			Iref = 1 / tau_refill[i] * (ions_NSR[i + idx * nions] - ions[idx_dyad]);
			double newc = ions[idx_dyad] + dt / beta * (Iref - IjSR_dyad[idx + i * nthreads] / V);
			int idx_ion = 0;
			for (int j = 0; j < nchannels; ++j) {
				idx_ion = _GetIdx(i, j, idx);
				ions_new[idx_ion] = newc;
			}
		}
	}
}

std::map < std::string, std::vector<double> > SRPool::GetConcentrations(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < nions; ++i) {
		auto it = std::find(values.begin(), values.end(), ion_names[i]);
		if (it != values.end()) {
			out.insert(std::pair<std::string, std::vector<double> >(ion_names[i], 
				std::vector<double>(ions.begin() + i * nthreads * nchannels, ions.begin() + (i+1) * nthreads * nchannels)));
		}
	}
	return out;
}

void SRPool::Reset() {
	for (int i = 0; i < nions; ++i) {
		for (int j = 0; j < nchannels; ++j) {
			for (int k = 0; k < nthreads; ++k) {
				long idx = _GetIdx(i, j, k);
				ions[idx] = ions_0[i];
				ions_new[idx] = ions_0[i];
			}
		}
	}
	for (int i = 0; i < nions; ++i)
		for (int k = 0; k < nthreads; ++k) 
			IjSR_dyad[k + i * nthreads] = 0;
}

bool SRPool::UsesGPU() {
	return false;
}

std::vector<uint64_t> SRPool::GetDimensions() {
	return sr_dims;
}
