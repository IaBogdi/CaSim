#include "LinearNaraghiNeher.h"

#include <Eigen/Eigen/EigenValues>
#include <Eigen/Eigen/Dense>
#include <iostream>

using namespace Eigen;

LinearNaraghiNeher::LinearNaraghiNeher(std::map < std::string, std::map< std::string, double> >& parameters, std::vector<std::shared_ptr < Point > >& points, double current,double ca_0,double dca,double ca_open) {
	std::vector<double> k, tau, d;
	background = ca_0;
	for (auto const& imap : parameters) {
		auto buffer = imap.first;
		auto kon = parameters[buffer]["kon"];
		auto koff = parameters[buffer]["koff"];
		auto D = parameters[buffer]["D"];
		auto conc = parameters[buffer]["Total Concentration"];
		double K = koff / kon;
		k.push_back(conc * K / ((K + ca_0) * (K + ca_0)));
		tau.push_back(1 / (koff + kon * ca_0));
		d.push_back(D);
	}
	MatrixXd Dm = MatrixXd::Zero(d.size() + 1, d.size() + 1);
	for (int i = 0; i < d.size(); ++i)
		Dm(i, i) = d[i];
	Dm(d.size(), d.size()) = dca;
	MatrixXd AA = MatrixXd::Zero(k.size() + 1, k.size() + 1);
	for (int i = 0; i < k.size(); ++i) {
		AA(i, i) = -1 / tau[i];
		AA(i, k.size()) = k[i] / tau[i];
		AA(k.size(), i) = -AA(i, i);
		AA(k.size(), k.size()) -= AA(i, k.size());
	}
	MatrixXd B = -Dm.inverse() * AA;
	EigenSolver<MatrixXd> es;
	es.compute(B);
	VectorXd u = VectorXd::Zero(k.size() + 1);
	MatrixXd eigv(k.size() + 1, k.size() + 1);
	for (int i = 0; i < k.size() + 1; ++i)
		for (int j = 0; j < k.size() + 1; ++j)
			eigv(i, j) = real(es.eigenvectors().col(j)[i]);
	double min_eig = 1e9;
	int min_idx = -1;
	VectorXd mu(k.size() + 1);
	for (int i = 0; i < k.size() + 1; ++i) {
		mu(i) = real(es.eigenvalues().col(0)[i]);
		if (mu(i) < min_eig) {
			min_eig = mu(i);
			min_idx = i;
		}
	}
	mu(min_idx) += mu(k.size());
	mu(k.size()) = mu(min_idx) - mu(k.size());
	mu(min_idx) -= mu(k.size());
	eigv.col(min_idx).swap(eigv.col(k.size()));
	u(k.size()) = current / (4 * Pi * Far * dca);
	VectorXd aa = eigv.fullPivLu().solve(u);
	for (int i = 0; i < points.size(); ++i) {
		std::vector<double> concentrations;
		for (int j = 0; j < points.size(); ++j) {
			auto r = points[i]->d(*points[j]);
			double dCa = 0;
			if (r > 1e-6)  {
				for (int ii = 0; ii < k.size() + 1; ++ii) {
					dCa += 1 / r * aa(ii) * std::exp(-r * std::sqrt(std::abs(mu(ii))) * 1e-8) * eigv(k.size(), ii) * 1e-4;
				}
			}
			else {
				dCa = ca_open;
			}
			concentrations.push_back(dCa);
		}
		added_concentrations.push_back(concentrations);
	}
}
