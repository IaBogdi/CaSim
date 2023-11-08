#pragma once
#include "QMatrixModel.h"
class SingleSiteMWC : public QMatrixModel {
	double KMgI,
		konMgI,
		koffMgI,
		KCa,
		alphaCa,
		konCa,
		koffCa,
		KMg,
		konMg,
		koffMg,
		alphaMg,
		fCa,
		fCa4I,
		fMg,
		KO0,
		alphakco,
		alphakcoI,
		kco,
		koc,
		gMg,
		gCa,
		fICa,
		fIMg,
		alphaI,
		alphaMgI;
protected:
	void RebuildQMatrix(const std::unordered_map<std::string, double>&);
	void RebuildQMatrixofSingleState(const std::unordered_map<std::string, double>&);
private:
	void SetRow(int,double, int,double);
	void _BuildAdjacencyList();
	void _AddTwoStatesInfo(int, int);
public:
	SingleSiteMWC(std::map<std::string, double>&);
};

