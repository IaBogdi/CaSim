#pragma once
#include "TwoStateModel.h"
#include "spline.h"
#include <memory>
class TwoStateSplineModel : public TwoStateModel
{
private:
	std::unique_ptr<tk::spline> _kopen;
	std::unique_ptr<tk::spline> _kclose;
protected:
	double _KClose(const std::unordered_map<std::string, double>&);
	double _KOpen(const std::unordered_map<std::string, double>&);
public:
	TwoStateSplineModel(std::string&,std::string&);
};

