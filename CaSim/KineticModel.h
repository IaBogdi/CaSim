#pragma once
#include <vector>
#include <map>
#include <string>

class KineticModel {
protected:
	int cur_state;
public:
	virtual double GetRate(std::vector<double>&) = 0;
	int GetState() {
		return cur_state;
	}
	virtual bool isOpen() = 0;
	virtual void MakeTransition(double) = 0;
	virtual void SetMacroState(int,double,std::vector<double>&) = 0;
	virtual int GetMacroState() = 0;
	virtual int GetMacroState(int) = 0;
	virtual int GetNStates() = 0;
};

