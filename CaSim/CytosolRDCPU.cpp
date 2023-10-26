#include "CytosolRDCPU.h"

inline double CytosolRDCPU::d2YdX2(double Yl, double Yc, double Yr, double dX)
{
	return (Yr - 2 * Yc + Yl) / (dX * dX);
}

inline double CytosolRDCPU::dYdX(double Yl, double Yr, double dX) {
	return (Yr - Yl) / (2 * dX);
}

void CytosolRDCPU::Initialize(nlohmann::json& j, nlohmann::json& j_dyad, int nthreads) {
	dr = j["dr"];
	r = j["r"];
	n_threads = nthreads;
	nr = r / dr + 1;
	cytosol_dims.resize(3);
	cytosol_dims[1] = nr;
	cytosol_dims[2] = n_threads;
	R = j_dyad["Radius"];
	n_elements = nr * n_threads;
	n_ions = 0;
	n_ions_dyad = 0;
	//ions
	for (auto const& el : j_dyad["Ions"].items()) {
		auto pars_ion = el.value();
		ions_data.push_back(std::make_unique < Ion<double> >(el.key(), pars_ion["D"], pars_ion["Concentration"]));
		if (pars_ion["Name"] == "Calcium")
			ca_idx = n_ions;
		++n_ions;
		++n_ions_dyad;
		extracell_ions.push_back(pars_ion["Extracell"]);
		ions_bg.push_back(pars_ion["Concentration"]);
		D_ions.push_back(pars_ion["D"]);
	}
	for (auto const& el : j["Ions"].items()) {
		auto pars_ion = el.value();
		ions_data.push_back(std::make_unique<Ion<double> >(el.key(), pars_ion["D"], pars_ion["Concentration"]));
		++n_ions;
		extracell_ions.push_back(pars_ion["Extracell"]);
		ions_bg.push_back(pars_ion["Concentration"]);
		D_ions.push_back(pars_ion["D"]);
	}
	//Buffers
	int idx_buff_dyad = 0;
	for (auto const& el : j_dyad["Buffers"].items()) {
		auto pars_buff = el.value();
		buffers_data.push_back(std::make_unique<Buffer<double> >(el.key(), pars_buff["Total Concentration"], true));
		total_buf.push_back(pars_buff["Total Concentration"]);
		for (auto const& el_ion : pars_buff["Ions"].items()) {
			auto pars_buff_ion = el_ion.value();
			buffers_data[buffers_data.size() - 1]->ions_kinetics[el_ion.key()] = std::make_unique<BufferIon<double> >(pars_buff_ion["D"], pars_buff_ion["kon"], pars_buff_ion["koff"]);
			++idx_buff_dyad;
		}
		int n_ions_buf = buffers_data[buffers_data.size() - 1]->ions_kinetics.size();
		std::vector<double> zeroes(n_ions_buf + 1, 0);
		std::vector<std::vector<double> > a(n_ions_buf, zeroes);
		int i_temp = 0;
		for (auto const& i : ions_data) {
			auto it = buffers_data[buffers_data.size() - 1]->ions_kinetics.find(i->name);
			if (it != buffers_data[buffers_data.size() - 1]->ions_kinetics.end()) {
				if (j_dyad["Start"] == "Equilibrium") {
					for (int I = 0; I < n_ions_buf; ++I) {
						a[i_temp][I] = it->second->kon * i->Cb;
						if (I == i_temp)
							a[i_temp][I] += it->second->koff;
					}
					a[i_temp][n_ions_buf] = it->second->kon * i->Cb * buffers_data[buffers_data.size() - 1]->Ctot;
					++i_temp;
				}
			}
		}
		if (j_dyad["Start"] == "Equilibrium") {
			auto c = GaussElimination(a);
			int t_buf = 0;
			for (auto const& i : ions_data) {
				auto it = buffers_data[buffers_data.size() - 1]->ions_kinetics.find(i->name);
				if (it != buffers_data[buffers_data.size() - 1]->ions_kinetics.end()) {
					buffers_data[buffers_data.size() - 1]->ions_kinetics[it->first]->initial_C = c[t_buf];
					++t_buf;
				}
			}
		}
	}
	for (auto const& el : j["Buffers"].items()) {
		auto pars_buff = el.value();
		buffers_data.push_back(std::make_unique<Buffer<double> >(el.key(), pars_buff["Total Concentration"]));
		total_buf.push_back(pars_buff["Total Concentration"]);
		for (auto const& el_ion : pars_buff["Ions"].items()) {
			auto pars_buff_ion = el_ion.value();
			buffers_data[buffers_data.size() - 1]->ions_kinetics[el_ion.key()] = std::make_unique<BufferIon<double> >(pars_buff_ion["D"], pars_buff_ion["kon"], pars_buff_ion["koff"]);
		}
		int n_ions_buf = buffers_data[buffers_data.size() - 1]->ions_kinetics.size();
		std::vector<double> zeroes(n_ions_buf + 1, 0);
		std::vector<std::vector<double> > a(n_ions_buf, zeroes);
		int i_temp = 0;
		for (auto const& i : ions_data) {
			auto it = buffers_data[buffers_data.size() - 1]->ions_kinetics.find(i->name);
			if (it != buffers_data[buffers_data.size() - 1]->ions_kinetics.end()) {
				if (j_dyad["Start"] == "Equilibrium") {
					for (int I = 0; I < n_ions_buf; ++I) {
						a[i_temp][I] = it->second->kon * i->Cb;
						if (I == i_temp)
							a[i_temp][I] += it->second->koff;
					}
					a[i_temp][n_ions_buf] = it->second->kon * i->Cb * buffers_data[buffers_data.size() - 1]->Ctot;
					++i_temp;
				}
			}
		}
		if (j_dyad["Start"] == "Equilibrium") {
			auto c = GaussElimination(a);
			int t_buf = 0;
			for (auto const& i : ions_data) {
				auto it = buffers_data[buffers_data.size() - 1]->ions_kinetics.find(i->name);
				if (it != buffers_data[buffers_data.size() - 1]->ions_kinetics.end()) {
					buffers_data[buffers_data.size() - 1]->ions_kinetics[it->first]->initial_C = c[t_buf];
					++t_buf;
				}
			}
		}
	}
	n_buf_unique = buffers_data.size();
	idx_buf = 0;
	for (auto const& i : ions_data) {
		for (auto const& b : buffers_data) {
			auto it = b->ions_kinetics.find(i->name);
			if (it != b->ions_kinetics.end()) {
				buffer_binds_ion.push_back(1);
				D_buf.push_back(it->second->D);
				kon_buf.push_back(it->second->kon);
				koff_buf.push_back(it->second->koff);
				buf_init.push_back(it->second->initial_C);
				idx_bufion[b->name][i->name] = idx_buf;
				++idx_buf;
				if (b->is_in_dyad)
					is_in_dyad.push_back(1);
				else
					is_in_dyad.push_back(0);
			}
			else
				buffer_binds_ion.push_back(0);
		}
	}
	for (auto const& el : j["SERCA"].items()) {
		auto pars_SR = el.value();
		ions_and_SR[el.key()] = std::make_unique<IonSERCA<double> >(el.key(), pars_SR["Jmax"], pars_SR["Kup"]);
	}
	for (auto const& i : ions_data) {
		auto it = ions_and_SR.find(i->name);
		if (it != ions_and_SR.end()) {
			SR_uptakes_ion.push_back(1);
			Jmax.push_back(it->second->Jmax);
			Kup.push_back(it->second->Kup);
		}
		else
			SR_uptakes_ion.push_back(0);
	}
	n_buffers = idx_buf;
	dt_is_set = false;
	ions_boundary = new double[n_ions * n_threads];
	ions = new double[n_elements * n_ions];
	zero_vector.resize((n_ions_dyad + idx_buff_dyad) * n_threads);
	std::fill(zero_vector.begin(), zero_vector.end(), 0);
	_SetNumIonsandBuffsinDyad(n_ions_dyad, idx_buff_dyad);
	buffers_boundary = new double[idx_buf * n_threads];
	buffers = new double[n_elements * idx_buf];
	buf_free = new double[n_elements * n_buf_unique];
	evo_ions = new double[n_elements * n_ions];
	evo_buffers = new double[n_elements * idx_buf];
}

CytosolRDCPU::CytosolRDCPU(nlohmann::json& j, nlohmann::json& j_dyad, int nthreads) {
	Initialize(j, j_dyad, nthreads);
}

CytosolRDCPU::CytosolRDCPU(nlohmann::json& j, nlohmann::json& j_dyad, nlohmann::json& j_nsr, int nthreads) {
	for (auto const& el : j_nsr["Ions"].items()) {
		auto pars_ion = el.value();
		nsr_ions.push_back(pars_ion["Concentration (uM)"]);
	}
	Initialize(j, j_dyad, nthreads);
}

std::vector<double> CytosolRDCPU::GaussElimination(std::vector<std::vector<double> >& a) {
	int n = (int)a.size();
	int m = (int)a[0].size() - 1;
	std::vector<double> ans;
	std::vector<int> where(m, -1);

	for (int col = 0, row = 0; col < m && row < n; ++col) {
		int sel = row;
		for (int i = row; i < n; ++i)
			if (abs(a[i][col]) > abs(a[sel][col]))
				sel = i;
		if (abs(a[sel][col]) < 1e-10)
			continue;
		for (int i = col; i <= m; ++i)
			std::swap(a[sel][i], a[row][i]);
		where[col] = row;

		for (int i = 0; i < n; ++i)
			if (i != row) {
				double c = a[i][col] / a[row][col];
				for (int j = col; j <= m; ++j)
					a[i][j] -= a[row][j] * c;
			}
		++row;
	}
	ans.assign(m, 0);
	for (int i = 0; i < m; ++i)
		if (where[i] != -1)
			ans[i] = a[where[i]][m] / a[where[i]][i];
	return ans;
}

void CytosolRDCPU::_SetNumIonsandBuffsinDyad(int nions, int nbuf) {
	n_ions_dyad = nions;
	n_buffs_dyad = nbuf;
}

std::vector<double>& CytosolRDCPU::GetExtraCellularIons() {
	return extracell_ions;
}

int CytosolRDCPU::_Index(int idr, int idx_batch) {
	return idr + nr * idx_batch;
}

void CytosolRDCPU::Reset() {
	int idfull;
	for (int idr = 0; idr < nr; ++idr)
		for (int idx_batch = 0; idx_batch < n_threads; ++idx_batch) {
			idfull = _Index(idr, idx_batch);
			int _idx_buff = 0;
			for (int j = 0; j < n_ions; ++j) {
				int _idx_ion = idfull + j * n_elements;
				ions[_idx_ion] = ions_bg[j];
				for (int i = 0; i < n_buf_unique; ++i) {
					int idx_b = idfull + i * n_elements;
					if (j == 0) {
						buf_free[idx_b] = total_buf[i];
					}
					if (buffer_binds_ion[i + j * n_buf_unique]) {
						double t = buf_init[_idx_buff];
						buffers[idfull + _idx_buff * n_elements] = t;
						buf_free[idx_b] -= t;
						++_idx_buff;
					}
				}
			}
		}
}

void CytosolRDCPU::_EvolutionOp(int idr, int idx_batch, double dt) {
	int idfull = _Index(idr, idx_batch);
	int idrl = _Index(idr - 1, idx_batch);
	int idrr = _Index(idr + 1, idx_batch);
	double val_center;
	int str;
	//diffusion
	if (idr > 0) {
		idrl = idfull - 1;
		idrr = idfull + 1;
		double rr = dr * idr;
		int str;
		for (int j = 0; j < n_ions; ++j) {
			str = j * n_elements;
			val_center = ions[idfull + str];
			evo_ions[idfull + str] = D_ions[j] * (2 * dYdX(ions[idrl + str], ions[idrr + str], dr) / rr 
				+ d2YdX2(ions[idrl + str], val_center, ions[idrr + str], dr));
			for (int i = 0; i < n_buffers; ++i) {
				str = i * n_elements;
				val_center = buffers[idfull + str];
				evo_buffers[idfull + str] = D_buf[i] * (2 * dYdX(buffers[idrl + str], buffers[idrr + str], dr) / rr
					+ d2YdX2(buffers[idrl + str], val_center, buffers[idrr + str], dr));
			}
		}
	}
	else {
		int idrr = idfull + 1;
		int str;
		for (int j = 0; j < n_ions; ++j) {
			str = j * n_elements;
			val_center = ions[idfull + str];
			evo_ions[idfull + str] = D_ions[j] * (d2YdX2(ions[idrr + str], val_center, ions[idrr + str], dr));
			for (int i = 0; i < n_buffers; ++i) {
				str = i * n_elements;
				val_center = buffers[idfull + str];
				evo_buffers[idfull + str] = D_buf[i] * (d2YdX2(buffers[idrr + str], val_center, buffers[idrr + str], dr));
			}
		}
	}
	//reaction
	double react;
	int str2, str3;
	int _idx_buff = 0;
	double R1;
	for (int j = 0; j < n_ions; ++j) {
		str = idfull + j * n_elements;
		react = 0;
		for (int i = 0; i < n_buf_unique; ++i) {
			if (buffer_binds_ion[i + j * n_buf_unique]) {
				if (is_in_dyad[_idx_buff] || !(idr * dr <= R)) {
				str2 = idfull + _idx_buff * n_elements;
				str3 = idfull + i * n_elements;
				R1 = koff_buf[_idx_buff] * buffers[str2] - kon_buf[_idx_buff] * ions[str] * buf_free[str3];
				evo_buffers[str2] += -R1;
				react += R1;
				}
				++_idx_buff;
			}
		}
		evo_ions[str] += react;
	}
}

void CytosolRDCPU::_UpdateEvo(int idr, int idx_batch, double dt) {
	int idx_SERCA = 0;
	int idx2;
	int idfull = _Index(idr, idx_batch);
	for (int j = 0; j < n_ions; ++j) {
		idx2 = idfull + j * n_elements;
		if (idr * dr <= R) {
			evo_ions[idx2] += currents[j + idx_batch * (n_ions_dyad + n_buffs_dyad)];
		}
		if (SR_uptakes_ion[j]) {
			double ions2 = ions[idx2] * ions[idx2];
			double K2 = Kup[idx_SERCA] * Kup[idx_SERCA];
			double ions2b = ions_bg[idx_SERCA] * ions_bg[idx_SERCA];
			evo_ions[idx2] -= Jmax[idx_SERCA] * (ions2 / (ions2 + K2) - ions2b / (ions2b + K2));
			++idx_SERCA;
		}
		ions[idx2] += evo_ions[idx2] * dt;
	}
	int idx_cur = 0;
	for (int i = 0; i < n_buffers; ++i) {
		idx2 = idfull + i * n_elements;
		if (idr * dr <= R && is_in_dyad[idx_cur]) {
			evo_buffers[idx2] += currents[n_ions_dyad + idx_cur + idx_batch * (n_ions_dyad + n_buffs_dyad)];
			++idx_cur;
		}
		buffers[idx2] += evo_buffers[idx2] * dt;
	}
}

void CytosolRDCPU::_GetFreeBuffers(int idr, int idx_batch) {
	int _idx_buff = 0;
	int idfull = _Index(idr, idx_batch);
	for (int j = 0; j < n_ions; ++j) {
		int idx2 = idfull + j * n_elements;
		int idx3;
		if (j == 0) {
			for (int i = 0; i < n_buf_unique; ++i)
				buf_free[idx2 + i * n_elements] = buffers_data[i]->Ctot;
		}
		for (int i = 0; i < n_buf_unique; ++i) {
			if (buffer_binds_ion[i + j * n_buf_unique]) {
				idx2 = idfull + _idx_buff * n_elements;
				idx3 = idfull + i * n_elements;
				buf_free[idx3] -= buffers[idx2];
				++_idx_buff;
			}
		}
	}
}

void CytosolRDCPU::RunRD(double dt, int idx_b) {
	for (int idr = 0; idr < nr - 1; ++idr)
		for (int idx_batch = 0; idx_batch < n_threads; ++idx_batch) 
			_EvolutionOp(idr, idx_batch, dt);
	for (int idr = 0; idr < nr - 1; ++idr)
		for (int idx_batch = 0; idx_batch < n_threads; ++idx_batch) {
			_UpdateEvo(idr, idx_batch, dt);
			_GetFreeBuffers(idr, idx_batch);
		}
}

void CytosolRDCPU::_GiveIonsAndBuffers() {
	int idx;
	for (int idfull = 0; idfull < n_threads; ++idfull) {
		idx = R / dr;
		double alpha = (R - idx * dr) / dr;
		for (int i = 0; i < n_ions_dyad; ++i) {
			ions_boundary[i + n_ions_dyad * idfull] = (1 - alpha) * ions[idx + i * n_elements] + alpha * ions[idx + 1 + i * n_elements];
		}
		int _idx_buf = 0;
		for (int i = 0; i < n_buffers; ++i) {
			if (is_in_dyad[i]) {
				buffers_boundary[_idx_buf + n_buffs_dyad * idfull] = (1 - alpha) * buffers[idx + i * n_elements] + alpha * buffers[idx + 1 + i * n_elements];
				++_idx_buf;
			}
		}
	}
}

void CytosolRDCPU::Update(double*& dyad_currents, const std::vector<double>& nsr_ions) {
	if (dyad_currents != currents)
		currents = dyad_currents;
}

void CytosolRDCPU::GetIonsandBuffers(double*& ions_boundary_out, double*& buffers_boundary_out) {
	_GiveIonsAndBuffers();
	ions_boundary_out = ions_boundary;
	buffers_boundary_out = buffers_boundary;
}

bool CytosolRDCPU::UsesGPU() {
	return false;
}

std::vector<uint64_t> CytosolRDCPU::GetDimensions() {
	return cytosol_dims;
}

std::map < std::string, std::vector<double> > CytosolRDCPU::GetConcentrations(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions_data.size(); ++i) {
		auto it = std::find(values.begin(), values.end(), ions_data[i]->name);
		if (it != values.end())
			out.insert(std::pair<std::string, std::vector<double> >(ions_data[i]->name, std::vector<double>(ions + i*n_elements, ions + (i+1) * n_elements)));
	}
	for (int i = 0; i < buffers_data.size(); ++i) {
		auto it = std::find(values.begin(), values.end(), buffers_data[i]->name);
		if (it != values.end()) {
			auto buf_name = buffers_data[i]->name;
			for (auto const& ion : ions_data) {
				auto it2 = buffers_data[i]->ions_kinetics.find(ion->name);
				if (it2 != buffers_data[i]->ions_kinetics.end()) {
					auto ion_name = ion->name;
					auto type_name = ion_name + std::string("-") + buf_name;
					int idx_b = idx_bufion[buf_name][ion_name];
					out.insert(std::pair<std::string, std::vector<double> >(type_name, std::vector<double>(buffers + i * n_elements, buffers + (i + 1) * n_elements)));
				}
			}
		}
	}
	return out;
}

CytosolRDCPU::~CytosolRDCPU() {
	delete[] ions_boundary;
	delete[] ions;
	delete[] buffers_boundary;
	delete[] buffers;
	delete[] evo_ions;
	delete[] evo_buffers;
	delete[] buf_free;
}