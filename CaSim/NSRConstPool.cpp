#include "NSRConstPool.h"

NSRConstPool::NSRConstPool(nlohmann::json& j, int n_threads)
{
	for (auto const& el : j["Ions"].items()) {
		auto pars_ion = el.value();
		ions.push_back(pars_ion["Concentration (uM)"]);
	}
	for (auto ion : ions)
		for (int i = 0; i < n_threads; ++i) {
			ions_out.push_back(ion);
		}
}

bool NSRConstPool::UsesGPU() {
	return false;
}

std::vector<double> NSRConstPool::GetIons() {
	return ions_out;
}

void NSRConstPool::Update(double*& arr1, const std::vector<double>& arr2) {
	return;
}

void NSRConstPool::Reset() {
	return;
}
