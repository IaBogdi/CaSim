#pragma once
#include "KineticModel.h"
#include <map>
class QMatrixModel : public KineticModel
{
protected:
	std::vector<std::vector<double> > Q;
	std::vector<int> states;
	std::vector<std::vector<int> > adjacency_list;
	virtual void RebuildQMatrix(const std::unordered_map<std::string, double>&) = 0;
	virtual void RebuildQMatrixofSingleState(const std::unordered_map<std::string, double>&) = 0;
public:
	QMatrixModel(std::map<std::string, double>& pars);
	QMatrixModel();
	double GetRate(const std::unordered_map<std::string, double>&);
	bool isOpen();
	void MakeTransition(double);
	void SetMacroState(int, double, const std::unordered_map<std::string, double>&);
	int GetMacroState();
	int GetMacroState(int);
	int GetNStates();
};

