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
	double _KClose(std::vector<double>&);
	double _KOpen(std::vector<double>&);
public:
	TwoStateSplineModel(std::string&,std::string&);
};

