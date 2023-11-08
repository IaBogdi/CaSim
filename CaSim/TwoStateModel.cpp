#include "TwoStateModel.h"


double TwoStateModel::GetRate(const std::unordered_map<std::string, double>& ions) {
	if (cur_state)
		return _KClose(ions);
	return _KOpen(ions);
}

bool TwoStateModel::isOpen() {
	return cur_state;
}

void TwoStateModel::MakeTransition(double rand) {
	cur_state = (cur_state + 1) % 2;
}

void TwoStateModel::SetMacroState(int macro_state, double rand, const std::unordered_map<std::string, double>& ions) {
	cur_state = macro_state;
}

int TwoStateModel::GetMacroState() {
	return cur_state;
}

int TwoStateModel::GetMacroState(int state) {
	if (state < 0 || state >= 2)
		return -1;
	return state;
}

int TwoStateModel::GetNStates() {
	return 2;
}