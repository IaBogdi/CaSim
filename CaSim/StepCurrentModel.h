#pragma once
#include "KineticModel.h"
class StepCurrentModel : public KineticModel
{
	double T;
public:
	StepCurrentModel(double);
	StepCurrentModel();
	double GetRate(std::vector<double>&);
	bool isOpen();
	void MakeTransition(double);
	void SetMacroState(int, double, std::vector<double>&);
	int GetMacroState();
	int GetMacroState(int);
	int GetNStates();
};

