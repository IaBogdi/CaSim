#pragma once
#include "KineticModel.h"
class TwoStateModel :
    public KineticModel
{
protected:
	virtual double _KClose(std::vector<double>&) = 0;
	virtual double _KOpen(std::vector<double>&) = 0;
public:
	double GetRate(std::vector<double>& ions);
	bool isOpen();
	void MakeTransition(double rand);
	void SetMacroState(int, double, std::vector<double>&);
	int GetMacroState();
	int GetMacroState(int);
	int GetNStates();
};

