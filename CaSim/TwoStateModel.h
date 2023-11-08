#pragma once
#include "KineticModel.h"
#include <map>
class TwoStateModel :
    public KineticModel
{
protected:
	virtual double _KClose(const std::unordered_map<std::string, double>&) = 0;
	virtual double _KOpen(const std::unordered_map<std::string, double>&) = 0;
public:
	double GetRate(const std::unordered_map<std::string, double>& ions);
	bool isOpen();
	void MakeTransition(double rand);
	void SetMacroState(int, double, const std::unordered_map<std::string, double>&);
	int GetMacroState();
	int GetMacroState(int);
	int GetNStates();
};

