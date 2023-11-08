#include "StepCurrentModel.h"

double StepCurrentModel::GetRate(const std::unordered_map<std::string, double>& ions) {
	T -= ions.at("dt");
	return T - ions.at("dt");
}

StepCurrentModel::StepCurrentModel(double t) {
	T = t;
	cur_state = 0;
}

StepCurrentModel::StepCurrentModel() {
	T = 1;
	cur_state = 0;
}

bool StepCurrentModel::isOpen() {
	return cur_state;
}

void StepCurrentModel::MakeTransition(double rand) {
	cur_state = (cur_state + 1) % 2;
}

void StepCurrentModel::SetMacroState(int macro_state, double rand, const std::unordered_map<std::string, double>& ions) {
	cur_state = macro_state;
}

int StepCurrentModel::GetMacroState() {
	return cur_state;
}

int StepCurrentModel::GetMacroState(int state) {
	if (state < 0 || state >= 2)
		return -1;
	return state;
}

int StepCurrentModel::GetNStates() {
	return 2;
}