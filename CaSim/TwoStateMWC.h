#pragma once
#include "TwoStatePoModel.h"


class TwoStateMWC :public TwoStatePoModel {
protected:
	double Po(std::vector<double>&);
private:
	double fCa, 
		KCa, 
		fMg,
		KMg,
		KO0,
		KMgI;
public:
	TwoStateMWC(std::map<std::string, double>&);
};

