#include "SingleSiteMWC.h"
#include <cmath>
#include <iostream>

SingleSiteMWC::SingleSiteMWC(std::map<std::string, double>& params) {
	fCa4I = params["fCa4I"];
	fCa = params["fCa"];
	KCa = params["KCa"];
	alphaCa = params["alphaCa"];
	konCa = params["konCa"];
	koffCa = KCa * konCa;

	fMg = params["fMg"];
	KMg = params["KMg"];
	konMg = params["konMg"];
	koffMg = konMg * KMg;
	alphaMg = params["alphaMg"];
	alphaMgI = params["alphaMgI"];

	KO0 = params["KO0"];
	alphakco = params["alphakco"];
	alphakcoI = params["alphakcoI"];
	kco = params["kco"];
	koc = KO0 * kco;

	KMgI = params["KMgI"];
	konMgI = params["konmgI"];
	koffMgI = KMgI * KMgI * konMgI;


	fICa = params["fICa"];
	fIMg = params["fIMg"];
	gMg = params["gMg"];
	gCa = params["gCa"];
	alphaI = params["alphaI"];
	states.resize(60);
	for (int i = 0; i < 60; ++i) {
		if (i < 45)
			states[i] = 0;
		else
			states[i] = 1;
	}
	Q.resize(60,std::vector<double>(60,0));
	adjacency_list.resize(60);
	_BuildAdjacencyList();
}

void SingleSiteMWC::_AddTwoStatesInfo(int x, int y) {
	adjacency_list[x].push_back(y);
	adjacency_list[x + 45].push_back(y + 45);
	adjacency_list[x + 15].push_back(y + 15);
	adjacency_list[x + 30].push_back(y + 30);
}

void SingleSiteMWC::_BuildAdjacencyList() {
	int idx_c, idx_c2;
	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5 - i; ++j) {
			idx_c = 15 - int((6 - j) * (5 - j) / 2) + i;
			if (i + j < 4) {
				idx_c2 = 15 - int((6 - j) * (5 - j) / 2) + i + 1;
				_AddTwoStatesInfo(idx_c, idx_c2);
			}
			if (i > 0) {
				idx_c2 = 15 - int((6 - j) * (5 - j) / 2) + i - 1;
				_AddTwoStatesInfo(idx_c, idx_c2);
			}
			if (i + j < 4) {
				idx_c2 = 15 - int((6 - (j + 1)) * (5 - (j + 1)) / 2) + i;
				_AddTwoStatesInfo(idx_c, idx_c2);
			}
			if (j > 0) {
				idx_c2 = 15 - int((6 - (j - 1)) * (5 - (j - 1)) / 2) + i;
				_AddTwoStatesInfo(idx_c, idx_c2);
			}
			adjacency_list[idx_c].push_back(idx_c + 15);
			adjacency_list[idx_c + 15].push_back(idx_c);
			adjacency_list[idx_c + 45].push_back(idx_c + 30);
			adjacency_list[idx_c + 30].push_back(idx_c + 45);
			adjacency_list[idx_c].push_back(idx_c + 45);
			adjacency_list[idx_c + 45].push_back(idx_c);
			adjacency_list[idx_c + 15].push_back(idx_c + 30);
			adjacency_list[idx_c + 30].push_back(idx_c + 15);
		}
}

void SingleSiteMWC::SetRow(int i,double ca, int j,double mg) { 
	double ff = std::pow(fCa, i) * std::pow(fMg, j);
	double fIO = std::pow(fICa / fCa * gCa, i) * std::pow(fIMg / fMg * gMg, j);
	double fIC = std::pow(gCa, i) * std::pow(gMg, j);
	double fIco = std::pow(fICa, i) * std::pow(fIMg, j);
	int idx_c = 15 - (6 - j) * (5 - j) / 2 + i;
	int idx_c2;
	if (i + j < 4) { //calcium binding
		idx_c2 = 15 - (6 - j) * (5 - j) / 2 + i + 1;
		Q[idx_c][idx_c2] = (4 - i - j) * konCa * ca;
		Q[idx_c + 45][idx_c2 + 45] = (4 - i - j) * konCa * ca / std::pow(fCa, alphaCa);
		Q[idx_c + 15][idx_c2 + 15] = (4 - i - j) * konCa * ca;
		if (i == 3)
			Q[idx_c + 30][idx_c2 + 30] = (4 - i - j) * konCa * ca / std::pow(fCa4I, alphaCa);
		else
			Q[idx_c + 30][idx_c2 + 30] = (4 - i - j) * konCa * ca / std::pow(fICa, alphaCa);
	}
	if (i > 0) { //calcium unbinding
		int idx_c2 = 15 - (6 - j) * (5 - j) / 2 + i - 1;
		Q[idx_c][idx_c2] = i * koffCa;
		Q[idx_c + 45][idx_c2 + 45] = i * koffCa * std::pow(fCa, 1 - alphaCa);
		Q[idx_c + 15][idx_c2 + 15] = i * koffCa * gCa;
		if (i == 4)
			Q[idx_c + 30][idx_c2 + 30] = i * koffCa * std::pow(fCa4I, 1 - alphaCa) * gCa;
		else
			Q[idx_c + 30][idx_c2 + 30] = i * koffCa * std::pow(fICa, 1 - alphaCa) * gCa;
	}
	if (i + j < 4) { //magnesium binding
		int idx_c2 = 15 - (6 - (j + 1)) * (5 - (j + 1)) / 2 + i;
		Q[idx_c][idx_c2] = (4 - i - j) * konMg * mg;
		Q[idx_c + 45][idx_c2 + 45] = (4 - i - j) * konMg * mg / std::pow(fMg, alphaMg);
		Q[idx_c + 15][idx_c2 + 15] = (4 - i - j) * konMg * mg / std::pow(gMg, alphaMgI);
		Q[idx_c + 30][idx_c2 + 30] = (4 - i - j) * konMg * mg / (std::pow(fIMg, alphaMg) * std::pow(gMg, alphaMgI));
	}
	if (j > 0) { //magnesium unbinding
		int idx_c2 = 15 - (6 - (j - 1)) * (5 - (j - 1)) / 2 + i;
		Q[idx_c][idx_c2] = j * koffMg;
		Q[idx_c + 45][idx_c2 + 45] = j * koffMg * std::pow(fMg, 1 - alphaMg);
		Q[idx_c + 15][idx_c2 + 15] = j * koffMg * std::pow(gMg, 1 - alphaMgI);
		Q[idx_c + 30][idx_c2 + 30] = j * koffMg * std::pow(fIMg, 1 - alphaMg) * std::pow(gMg, 1 - alphaMgI);
	}
	Q[idx_c][idx_c + 15] = konMgI * std::pow(mg, 2) / std::pow(gMg, j);
	Q[idx_c + 15][idx_c] = koffMgI * std::pow(gCa, i);
	Q[idx_c + 45][idx_c + 30] = konMgI * std::pow(mg, 2) / std::pow(fIMg / fMg * gMg, j);
	if (i == 4)
		Q[idx_c + 30][idx_c + 45] = koffMgI * std::pow(fICa / fCa * gCa, 3) * fCa4I / fCa;
	else
		Q[idx_c + 30][idx_c + 45] = koffMgI * std::pow(fICa / fCa * gCa, i);
	Q[idx_c][idx_c + 45] = kco / std::pow(ff, alphakco);
	Q[idx_c + 45][idx_c] = koc * std::pow(ff, 1 - alphakco);
	if (i == 4)
		fIco = std::pow(fICa, 3) * fCa4I;
	Q[idx_c + 15][idx_c + 30] = kco / std::pow(fIco, alphakcoI);
	Q[idx_c + 30][idx_c + 15] = koc * std::pow(fIco, 1 - alphakcoI);
}

void SingleSiteMWC::RebuildQMatrixofSingleState(std::vector<double>& ions) {
	double ca = ions[0];
	double mg = ions[1];
	int this_state = cur_state % 15;
	for (int j = 0; j <= 4; ++j) {
		if (this_state - (5 - j) < 0) {
			SetRow(this_state, ca, j, mg);
			double s = 0;
			for (int k = 0; k < adjacency_list[cur_state].size(); ++k)
				s += Q[cur_state][adjacency_list[cur_state][k]];
			Q[cur_state][cur_state] = -s;
			return;
		}
		else
			this_state -= 5 - j;
	}
}

void SingleSiteMWC::RebuildQMatrix(std::vector<double>& ions) {
	double ca = ions[0];
	double mg = ions[1];
	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5 - i; ++j) {
			SetRow(i, ca, j, mg);
		}
	}
	for (int i = 0; i < states.size(); ++i) {
		double s = 0;
		for (int j = 0; j < adjacency_list[i].size(); ++j)
			s += Q[i][adjacency_list[i][j]];
		Q[i][i] = -s;
	}
}

