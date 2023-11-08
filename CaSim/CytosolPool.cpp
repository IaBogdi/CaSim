#include "CytosolPool.h"

void CytosolPool::Initialize(nlohmann::json& j, nlohmann::json& j_dyad, int nthreads) {
	n_threads = nthreads;
	cytosol_dims.resize(2);
	cytosol_dims[1] = n_threads;
	R = j_dyad["Radius"];
	n_ions = 0;
	n_ions_dyad = 0;
	onetau_coeff = j["Leak mult"];
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
		onetau_ions.push_back(onetau_coeff * pars_ion["D"]);
	}
	for (auto const& el : j["Ions"].items()) {
		auto pars_ion = el.value();
		ions_data.push_back(std::make_unique<Ion<double> >(el.key(), pars_ion["D"], pars_ion["Concentration"]));
		++n_ions;
		extracell_ions.push_back(pars_ion["Extracell"]);
		ions_bg.push_back(pars_ion["Concentration"]);
		onetau_ions.push_back(onetau_coeff * pars_ion["D"]);
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
	n_buffers = 0;
	for (auto const& i : ions_data) {
		for (auto const& b : buffers_data) {
			auto it = b->ions_kinetics.find(i->name);
			if (it != b->ions_kinetics.end()) {
				buffer_binds_ion.push_back(1);
				kon_buf.push_back(it->second->kon);
				koff_buf.push_back(it->second->koff);
				buf_init.push_back(it->second->initial_C);
				onetau_buffers.push_back(onetau_coeff *  it->second->D);
				idx_bufion[b->name][i->name] = n_buffers;
				++n_buffers;
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
	V = j_dyad["Voltage"];
	dt_is_set = false;
	ions = new double[n_threads * n_ions];
	zero_vector.resize((n_ions_dyad + idx_buff_dyad) * n_threads);
	std::fill(zero_vector.begin(), zero_vector.end(), 0);
	_SetNumIonsandBuffsinDyad(n_ions_dyad, idx_buff_dyad);
	buffers_boundary = new double[n_threads * idx_buff_dyad];
	buffers = new double[n_threads * n_buffers];
	buf_free = new double[n_threads * n_buf_unique];
	evo_ions = new double[n_threads * n_ions];
	evo_buffers = new double[n_threads * n_buffers];
}

CytosolPool::CytosolPool(nlohmann::json& j, nlohmann::json& j_dyad, int nthreads) {
	Initialize(j, j_dyad, nthreads);
}

CytosolPool::CytosolPool(nlohmann::json& j, nlohmann::json& j_dyad, nlohmann::json& j_nsr, int nthreads) {
	for (auto const& el : j_nsr["Ions"].items()) {
		auto pars_ion = el.value();
		nsr_ions.push_back(pars_ion["Concentration (uM)"]);
	}
	Initialize(j, j_dyad, nthreads);
}

std::vector<double> CytosolPool::GaussElimination(std::vector<std::vector<double> >& a) {
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

void CytosolPool::_SetNumIonsandBuffsinDyad(int nions, int nbuf) {
	n_ions_dyad = nions;
	n_buffs_dyad = nbuf;
}

std::vector<double>& CytosolPool::GetExtraCellularIons() {
	return extracell_ions;
}

void CytosolPool::Update(double*& dyad_currents, const std::vector<double>& nsr_ions) {
	if (dyad_currents != currents)
		currents = dyad_currents;
}

void CytosolPool::RunRD(double dt, int idx) {
	//Calculate reaction + sources
	double react;
	int str, str2, str3;
	int _idx_buff = 0;
	int idx_buf_dyad = 0;
	double R1;
	int idx_SERCA = 0;
	for (int j = 0; j < n_ions; ++j) {
		react = 0;
		str = idx + j * n_threads;
		for (int i = 0; i < n_buf_unique; ++i) {
			if (buffer_binds_ion[i + j * n_buf_unique]) {
				str2 = idx + _idx_buff * n_threads;
				str3 = idx + i * n_threads;
				R1 = koff_buf[_idx_buff] * buffers[str2] - kon_buf[_idx_buff] * ions[str] * buf_free[str3];
				evo_buffers[str2] = -R1;
				react += R1;
				if (is_in_dyad[_idx_buff]) {
					evo_buffers[str2] += currents[n_ions + idx_buf_dyad + idx * (n_ions_dyad + n_buffs_dyad)];
					++idx_buf_dyad;
				}
				++_idx_buff;
			}
		}
		evo_ions[str] = react + currents[j + idx * (n_ions_dyad + n_buffs_dyad)];
		if (SR_uptakes_ion[j]) {
			double ions2 = ions[str] * ions[str];
			double K2 = Kup[idx_SERCA] * Kup[idx_SERCA];
			double ions2b = ions_bg[idx_SERCA] * ions_bg[idx_SERCA];
			evo_ions[str] -= Jmax[idx_SERCA] * (ions2 / (ions2 + K2) - ions2b / (ions2b + K2));
			++idx_SERCA;
		}
	}
	//calculate next concentration
	for (int j = 0; j < n_ions; ++j) {
		str = idx + j * n_threads;
		evo_ions[str] -= onetau_ions[j] * (ions[str] - ions_bg[j]); //leak term
		ions[str] += evo_ions[str] * dt;
	}
	idx_buf_dyad = 0;
	for (int i = 0; i < n_buffers; ++i) {
		str = idx + i * n_threads;
		evo_buffers[str] -= onetau_buffers[i] * (buffers[str] - buf_init[i]); //leak term
		buffers[str] += evo_buffers[str] * dt;
		if (is_in_dyad[i]) {
			buffers_boundary[idx + idx_buf_dyad * n_threads] = buffers[str];
			++idx_buf_dyad;
		}
	}
	_GetFreeBuffers(idx);
}

void CytosolPool::_GetFreeBuffers(int idx_batch) {
	int _idx_buff = 0;
	for (int j = 0; j < n_ions; ++j) {
		int idx2 = idx_batch + j * n_threads;
		int idx3;
		if (j == 0) {
			for (int i = 0; i < n_buf_unique; ++i)
				buf_free[idx_batch + i * n_threads] = buffers_data[i]->Ctot;
		}
		for (int i = 0; i < n_buf_unique; ++i) {
			if (buffer_binds_ion[i + j * n_buf_unique]) {
				idx2 = idx_batch + _idx_buff * n_threads;
				idx3 = idx_batch + i * n_threads;
				buf_free[idx3] -= buffers[idx2];
				++_idx_buff;
			}
		}
	}
}

void CytosolPool::Reset() {
	for (int idx_batch = 0; idx_batch < n_threads; ++idx_batch) {
		int _idx_buff = 0;
		int _idx_buff_dyad = 0;
		for (int j = 0; j < n_ions; ++j) {
			int _idx_ion = idx_batch + j * n_threads;
			ions[_idx_ion] = ions_bg[j];
			for (int i = 0; i < n_buf_unique; ++i) {
				int idx_b = idx_batch + i * n_threads;
				if (j == 0) {
					buf_free[idx_b] = total_buf[i];
				}
				if (buffer_binds_ion[i + j * n_buf_unique]) {
					double t = buf_init[_idx_buff];
					buffers[idx_batch + _idx_buff * n_threads] = t;
					if (is_in_dyad[_idx_buff]) {
						buffers_boundary[idx_batch + _idx_buff_dyad * n_threads] = t;
						++_idx_buff_dyad;
					}
					buf_free[idx_b] -= t;
					++_idx_buff;
				}
			}
		}
	}
}

void CytosolPool::GetIonsBuffersandV(double*& ions_boundary_out, double*& buffers_boundary_out,double& Vol) {
	ions_boundary_out = ions;
	buffers_boundary_out = buffers_boundary;
	Vol = V;
}

bool CytosolPool::UsesGPU() {
	return false;
}

std::vector<uint64_t> CytosolPool::GetDimensions() {
	return cytosol_dims;
}

std::map < std::string, std::vector<double> > CytosolPool::GetConcentrations(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions_data.size(); ++i) {
		auto it = std::find(values.begin(), values.end(), ions_data[i]->name);
		if (it != values.end()) {
			out.insert(std::pair<std::string, std::vector<double> >(ions_data[i]->name, std::vector<double>(ions + i * n_threads, ions + (i + 1) * n_threads)));
			auto a = out[ions_data[i]->name];
		}
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
					out.insert(std::pair<std::string, std::vector<double> >(type_name, std::vector<double>(buffers + idx_b * n_threads, buffers + (idx_b + 1) * n_threads)));
				}
			}
		}
	}
	return out;
}

CytosolPool::~CytosolPool() {
	delete[] ions;
	delete[] buffers;
	delete[] evo_ions;
	delete[] evo_buffers;
	delete[] buf_free;
	delete[] buffers_boundary;
}