#pragma once
#include "KineticModel.h"
class StepCurrentModel : public KineticModel
{
	double T;
public:
	StepCurrentModel(double);
	StepCurrentModel();
	double GetRate(const std::unordered_map<std::string, double>&);
	bool isOpen();
	void MakeTransition(double);
	void SetMacroState(int, double, const std::unordered_map<std::string, double>&);
	int GetMacroState();
	int GetMacroState(int);
	int GetNStates();
};

