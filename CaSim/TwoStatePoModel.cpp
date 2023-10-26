#include "TwoStatePoModel.h"

double TwoStatePoModel::GetRate(std::vector<double>& ions) {
	if (cur_state)
		return kclose;
	double P = Po(ions);
	return kclose * P / (1 - P);
}

TwoStatePoModel::TwoStatePoModel(std::map<std::string, double>& params) {
	kclose = params["kclose"];
	cur_state = 0;
}

TwoStatePoModel::TwoStatePoModel() {
	kclose = 1;
	cur_state = 0;
}

bool TwoStatePoModel::isOpen() {
	return cur_state;
}

void TwoStatePoModel::MakeTransition(double rand) {
	cur_state = (cur_state + 1) % 2;
}

void TwoStatePoModel::SetMacroState(int macro_state, double rand, std::vector<double>& ions) {
	cur_state = macro_state;
}

int TwoStatePoModel::GetMacroState() {
	return cur_state;
}

int TwoStatePoModel::GetMacroState(int state) {
	if (state < 0 || state >= 2)
		return -1;
	return state;
}

int TwoStatePoModel::GetNStates() {
	return 2;
}
