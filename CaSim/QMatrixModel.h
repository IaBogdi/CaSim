#pragma once
#include "KineticModel.h"
class QMatrixModel : public KineticModel
{
protected:
	std::vector<std::vector<double> > Q;
	std::vector<int> states;
	std::vector<std::vector<int> > adjacency_list;
	virtual void RebuildQMatrix(std::vector<double>& ions) = 0;
	virtual void RebuildQMatrixofSingleState(std::vector<double>& ions) = 0;
public:
	QMatrixModel(std::map<std::string, double>& pars);
	QMatrixModel();
	double GetRate(std::vector<double>&);
	bool isOpen();
	void MakeTransition(double);
	void SetMacroState(int, double, std::vector<double>&);
	int GetMacroState();
	int GetMacroState(int);
	int GetNStates();
};

