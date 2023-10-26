#include "StepCurrentModel.h"

double StepCurrentModel::GetRate(std::vector<double>& ions) {
	T -= ions[0];
	return T - ions[0];
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

void StepCurrentModel::SetMacroState(int macro_state, double rand, std::vector<double>& ions) {
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