#include "QMatrixModel.h"

#include <Eigen/Eigen/Dense>
#include <iostream>

using namespace Eigen;

QMatrixModel::QMatrixModel(std::map<std::string, double>& pars) {
    cur_state = 0;
}

QMatrixModel::QMatrixModel() {
    cur_state = 0;
}

int QMatrixModel::GetNStates() {
    return states.size();
}

double QMatrixModel::GetRate(const std::unordered_map<std::string, double>& ions)
{
    RebuildQMatrixofSingleState(ions);
    return -Q[cur_state][cur_state];
}

void QMatrixModel::MakeTransition(double rand) {
    double s = 0;
    for (int i = 0; i < adjacency_list[cur_state].size(); ++i) {
        s += Q[cur_state][adjacency_list[cur_state][i]];
        if (-s / Q[cur_state][cur_state] >= rand) {
            cur_state = adjacency_list[cur_state][i];
            return;
        }
    }
}

void QMatrixModel::SetMacroState(int macro_state, double rand, const std::unordered_map<std::string, double>& ions) {
    RebuildQMatrix(ions);
    //S = [Q | 1]; Pe = 1 * (S * S.T).I
    MatrixXd S = MatrixXd::Ones(Q.size(), Q.size() + 1);
    for (int i = 0; i < Q.size(); ++i)
        for (int j = 0; j < Q.size(); ++j)
            S(i, j) = Q[i][j];
    RowVectorXd u = RowVectorXd::Ones(Q.size());
    VectorXd p = (u * (S * S.transpose()).inverse()).transpose();
    double total = 0;
    for (int i = 0; i < Q.size(); ++i) {
        total += (states[i] == macro_state) * p(i);
    }
    double s = 0;
    for (int i = 0; i < Q.size(); ++i) {
        s += (states[i] == macro_state) * p(i);
        if (s / total >= rand) {
            cur_state = i;
            return;
        }
    }
    return;
}

int QMatrixModel::GetMacroState() {
    return states[cur_state];
}

int QMatrixModel::GetMacroState(int state) {
    if (state < 0 || state >= states.size())
        return -1;
    return states[state];
}

bool QMatrixModel::isOpen() {
    return states[cur_state] > 0;
}