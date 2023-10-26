#include "DyadRDCPU.h"

#include <algorithm> //find
#include <utility>

inline double DyadRDCPU::d2YdX2(double Yl, double Yc, double Yr, double dX)
{
	return (Yr - 2 * Yc + Yl) / (dX * dX);
}

DyadRDCPU::DyadRDCPU(nlohmann::json& j, int nthreads) {
	channels = std::make_unique<DyadChannels>(j, nthreads);
	dx = j["dx"];
	dy = j["dy"];
	dz = j["dz"];
	x = j["x"];
	y = j["y"];
	z = j["z"];
	R = j["Radius"];
	n_threads = nthreads;
	nx = x / dx + 1;
	ny = y / dy + 1;
	nz = z / dz + 1;
	for (int i = 0; i < nz; ++i) {
		if (i == 0 || i == nz - 1)
			mult_z.push_back(0.5);
		else
			mult_z.push_back(1.0);
	}
	dyad_dims.resize(5);
	dyad_dims[1] = nx;
	dyad_dims[2] = ny;
	dyad_dims[3] = nz;
	dyad_dims[4] = n_threads;
	sx = dy * dz / dx;
	sy = dx * dz / dy;
	n_elements_thread = nx * ny * nz;
	n_elements = nx * ny * nz * n_threads;
	//Ions
	n_ions = 0;
	for (auto const& el : j["Ions"].items()) {
		auto pars_ion = el.value();
		ions_data.push_back(std::make_unique<Ion<double> >(el.key(), pars_ion["D"], pars_ion["Concentration"]));
		if (el.key() == "Calcium")
			ca_idx = n_ions;
		++n_ions;
		D_ions.push_back(pars_ion["D"]);
	}
	total_sr_current.resize(n_ions * n_threads);
	//Channels
	sl_size = channels->GetNSLIonChannels();
	sr_size = channels->GetNSRIonChannels();
	n_channels = sl_size + sr_size;
	x_idx = new int[n_channels];
	y_idx = new int[n_channels];
	z_idx = new int[n_channels];
	for (int j = 0; j < sr_size; ++j) {
		auto p = channels->GetSRIonChannel(0, j)->GetCoordinates()->GetCoordsOnGrid(dx, dy, dz);
		x_idx[j] = p[0];
		y_idx[j] = p[1];
		z_idx[j] = p[2];
	}
	for (int j = sr_size; j < n_channels; ++j) {
		auto p = channels->GetSLIonChannel(0, j - sr_size)->GetCoordinates()->GetCoordsOnGrid(dx, dy, dz);
		x_idx[j] = p[0];
		y_idx[j] = p[1];
		z_idx[j] = p[2];
	}
	n_elements_near_channels = n_ions * n_channels * n_threads;
	channels_ions_dims.resize(3);
	channels_ions_dims[1] = n_channels;
	channels_ions_dims[2] = n_threads;
	channels_dims.resize(3);
	channels_dims[1] = n_channels;
	channels_dims[2] = n_threads;
	//Buffers
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
				if (j["Start"] == "Equilibrium") {
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
		if (j["Start"] == "Equilibrium") {
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
	//loop ions and buffers so that the GPU computation would go smoothly
	idx_buf = 0;
	for (auto const& i : ions_data)
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
			}
			else
				buffer_binds_ion.push_back(0);
		}
	for (auto const& el : j["Sarcolemma"].items()) {
		auto pars_SL = el.value();
		ions_and_SL[el.key()] = std::make_unique<IonSarcolemma<double> >(pars_SL["N1"], pars_SL["K1"], pars_SL["N2"], pars_SL["K2"]);
	}
	for (auto const& i : ions_data) {
		auto it = ions_and_SL.find(i->name);
		if (it != ions_and_SL.end()) {
			SL_binds_ion.push_back(1);
			K1.push_back(it->second->K1);
			K2.push_back(it->second->K2);
			N1.push_back(it->second->N1);
			N2.push_back(it->second->N2);
		}
		else
			SL_binds_ion.push_back(0);
	}
	//calculate multiplier for gradient
	pointers_are_set = false;
	dt_is_set = false;
	grad_mult = 3.0f / (4 * acos(-1) * R * R * R);
	n_buffers = idx_buf;
	n_buf_unique = buffers_data.size();
	n_ions_and_buffers = idx_buf + n_ions;
	gradients = new double[(idx_buf + n_ions) * n_threads];
	ions.resize(n_elements * n_ions);
	currents = new double[n_elements * n_ions];
	buffers = new double[n_elements * idx_buf];
	evo_ions = new double[n_elements* n_ions];
	evo_buffers = new double[n_elements * idx_buf];
	init_ions = new double[n_ions]; 
	for (int i = 0; i < ions_data.size(); ++i)
		init_ions[i] = ions_data[i]->Cb;
	buf_free = new double[n_elements * n_buf_unique];
	ions_near_channels.resize(n_channels* n_threads* n_ions);
}

std::vector<double> DyadRDCPU::GaussElimination(std::vector<std::vector<double> >& a) {
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

double DyadRDCPU::GetElementVolume() {
	return dx * dy * dz;
}

int DyadRDCPU::GetNumSRChannels() {
	return sr_size;
}

void DyadRDCPU::Reset() {
	int idfull;
	for (int idx = 0; idx < nx; ++idx)
		for (int idy = 0; idy < ny; ++idy)
			for (int idz = 0; idz < nz; ++idz)
				for (int idx_batch = 0; idx_batch < n_threads; ++idx_batch) {
					idfull = _Index(idx, idy, idz, idx_batch);
					int _idx_buff = 0;
					for (int j = 0; j < n_ions; ++j) {
						int _idx_ion = idfull + j * n_elements;
						ions[_idx_ion] = init_ions[j];
						currents[_idx_ion] = 0;
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

bool DyadRDCPU::UsesGPU() {
	return false;
}

int DyadRDCPU::GetNIons() {
	return n_ions;
}

std::vector<double> DyadRDCPU::GetTotalSRCurrent() {
	return total_sr_current;
}

void DyadRDCPU::InitOpening(int thread, int channel) {
	channels->InitOpening(thread, channel);
}

void DyadRDCPU::_SetCurrents(const std::vector<double>& ions_jsr, const std::vector<double>& ions_extracell) {
	int idx, idx_jsr, idx_extra;
	for (int i = 0; i < n_threads; ++i) {
		total_sr_current[i] = 0;
		for (int j = 0; j < sr_size; ++j) {
			for (int k = 0; k < n_ions; ++k) {
				idx = j + n_channels * (i + k * n_threads);
				idx_jsr = j + sr_size * (i + k * n_threads);
				ions_near_channels[idx] = _GetIonNearChannel(j, i, k);
				currents[_GetChannelIndex(j,i,k)] = channels->GetSRIonChannel(i, j)->Flux(ions_jsr[idx_jsr], ions_near_channels[idx], k);
				total_sr_current[i + k * n_threads] += currents[idx];
			}
		}
		for (int j = sr_size; j < n_channels; ++j) {
			for (int k = 0; k < n_ions; ++k) {
				idx = j + n_channels * (i + k * n_threads);
				ions_near_channels[idx] = _GetIonNearChannel(j, i, k);
				idx_extra = j - sr_size + sl_size * (i + k * n_threads);
				currents[_GetChannelIndex(j, i, k)] = channels->GetSLIonChannel(i, j - sr_size)->Flux(ions_extracell[idx_extra], ions_near_channels[idx], k);
			}
		}
	}
}

void DyadRDCPU::GetEffluxes(double*& currents_out) {
	currents_out = gradients;
}

int DyadRDCPU::_Index(int idx, int idy, int idz, int idw) {
	return idx + nx * (idy + ny * (idz + nz * idw));
}

void DyadRDCPU::_SetBoundaries() {
	int idx = 0;
	int idy, idz, idx_batch, idfull, idx2;
	for (idx = 0; idx < nx; idx+= nx - 1)
		for (idy = 0; idy < ny; ++idy)
			for (idz = 0; idz < nz; ++idz)
				for (idx_batch = 0; idx_batch < n_threads; ++idx_batch) {
					idfull = _Index(idx, idy, idz, idx_batch);
					for (int j = 0; j < n_ions; ++j) {
						idx2 = idfull + j * n_elements;
						ions[idx2] = ions_from_cytosol[j + idx_batch * n_ions];
					}
					for (int i = 0; i < n_buffers; ++i) {
						idx2 = idfull + i * n_elements;
						buffers[idx2] = buffers_from_cytosol[i + idx_batch * n_buffers];
					}
				}
	for (idy = 0; idy < ny; idy += ny - 1)
		for (idx = 0; idx < nx; ++idx)
			for (idz = 0; idz < nz; ++idz)
				for (idx_batch = 0; idx_batch < n_threads; ++idx_batch) {
					idfull = _Index(idx, idy, idz, idx_batch);
					for (int j = 0; j < n_ions; ++j) {
						idx2 = idfull + j * n_elements;
						ions[idx2] = ions_from_cytosol[j + idx_batch * n_ions];
					}
					for (int i = 0; i < n_buffers; ++i) {
						idx2 = idfull + i * n_elements;
						buffers[idx2] = buffers_from_cytosol[i + idx_batch * n_buffers];
					}
				}
}

void DyadRDCPU::_GetGradients() {
	int idx, idy, idz;
	int n_batch;
	int str = 0;
	int idfull = 0;
	int idfull2 = 0;
	int idfull3 = 0;
	for (int j = 0; j < n_ions; ++j)
		for (n_batch = 0; n_batch < n_threads; ++n_batch) {
			gradients[j + n_ions_and_buffers * n_batch] = 0;
		}
	for (int i = 0; i < n_buffers; ++i)
		for (n_batch = 0; n_batch < n_threads; ++n_batch) {
			gradients[i + n_ions + n_ions_and_buffers * n_batch] = 0;
		}
	for (idx = 0; idx < nx; idx+=nx - 1)
		for (idy = 1; idy < ny - 1; ++idy)
			for (idz = 0; idz < nz; ++idz)
				for (n_batch = 0; n_batch < n_threads; ++n_batch) {
					idfull = idx == 0 ? _Index(idx + 1, idy, idz, n_batch) : _Index(idx - 1, idy, idz, n_batch);
					idfull2 = _Index(idx, idy, idz, n_batch);
					idfull3 = idx == 0 ? _Index(idx + 2, idy, idz, n_batch) : _Index(idx - 2, idy, idz, n_batch);
					for (int j = 0; j < n_ions; ++j) {
						str = j * n_elements;
						gradients[j + n_ions_and_buffers * n_batch] += mult_z[idz] * D_ions[j] * grad_mult * sx * (2 * ions[idfull + str] - 1.5 * ions[idfull2 + str] - 0.5 * ions[idfull3 + str]);
					}
					for (int i = 0; i < n_buffers; ++i) {
						str = i * n_elements;
						gradients[i + n_ions + n_ions_and_buffers * n_batch] += mult_z[idz] * D_buf[i] * grad_mult * sx * (2 * buffers[idfull + str] - 1.5 * buffers[idfull2 + str] - 0.5 * buffers[idfull3 + str]);
					}
				}
	for (idy = 0; idy < ny; idy += ny - 1)
		for (idx = 0; idx < nx; ++idx)
			for (idz = 0; idz < nz; ++idz)
				for (n_batch = 0; n_batch < n_threads; ++n_batch) {
					idfull = idy == 0 ? _Index(idx, idy + 1, idz, n_batch) : _Index(idx, idy - 1, idz, n_batch);
					idfull2 = _Index(idx, idy, idz, n_batch);
					idfull3 = idy == 0 ? _Index(idx, idy + 2, idz, n_batch) : _Index(idx, idy - 2, idz, n_batch);
					for (int j = 0; j < n_ions; ++j) {
						str = j * n_elements;
						gradients[j + n_ions_and_buffers * n_batch] += mult_z[idz] *  D_ions[j] * grad_mult * sy * (2 * ions[idfull + str] - 1.5 * ions[idfull2 + str] - 0.5 * ions[idfull3 + str]);
					}
					for (int i = 0; i < n_buffers; ++i) {
						str = i * n_elements;
						gradients[i + n_ions + n_ions_and_buffers * n_batch] += mult_z[idz] * D_buf[i] * grad_mult * sy * (2 * buffers[idfull + str] - 1.5 * buffers[idfull2 + str] - 0.5 * buffers[idfull3 + str]);
					}
				}
}

inline int DyadRDCPU::_GetChannelIndex(int idx_channel, int idx_thread, int idx_ion) {
	return _Index(x_idx[idx_channel], y_idx[idx_channel], z_idx[idx_channel], idx_thread) + idx_ion * n_elements;
}

inline double DyadRDCPU::_GetIonNearChannel(int idx_channel, int idx_thread, int idx_ion) {
	int id = _GetChannelIndex(idx_channel, idx_thread, idx_ion);
	return ions[id];
}

void DyadRDCPU::Update(double*& ions_cytosol, double*& buffers_cytosol, const std::vector<double>& jsr_ions, const std::vector<double>& extracellular_ions) {//add reinit
	if (ions_from_cytosol != ions_cytosol || buffers_from_cytosol != buffers_cytosol) {
		ions_from_cytosol = ions_cytosol;
		buffers_from_cytosol = buffers_cytosol;
	}
	_SetBoundaries();
	_GetGradients();
	_SetCurrents(jsr_ions, extracellular_ions);
}

void DyadRDCPU::_EvolutionOp(int idx, int idy, int idz, int idx_batch, double dt) {
	int idfull = _Index(idx, idy, idz, idx_batch);
	int idxl = _Index(idx - 1, idy, idz, idx_batch);
	int idxr = _Index(idx + 1, idy, idz, idx_batch);
	int idyl = _Index(idx, idy - 1, idz, idx_batch);
	int idyr = _Index(idx, idy + 1, idz, idx_batch);
	int idzl = _Index(idx, idy, idz - 1, idx_batch);
	int idzr = _Index(idx, idy, idz + 1, idx_batch);
	double val_center;
	int str, idx2;
	//diffusion
	if (idz == 0) {
		for (int j = 0; j < n_ions; ++j) {
			str = j * n_elements;
			val_center = ions[idfull + str];
			evo_ions[idfull + str] = D_ions[j] * (d2YdX2(ions[idxl + str], val_center, ions[idxr + str], dx) +
				d2YdX2(ions[idyl + str], val_center, ions[idyr + str], dy) +
				d2YdX2(ions[idzr + str], val_center, ions[idzr + str], dz));
		}
		for (int i = 0; i < n_buffers; ++i) {
			str = i * n_elements;
			val_center = buffers[idfull + str];
			evo_buffers[idfull + str] = D_buf[i] * (d2YdX2(buffers[idxl + str], val_center, buffers[idxr + str], dx) +
				d2YdX2(buffers[idyl + str], val_center, buffers[idyr + str], dy) +
				d2YdX2(buffers[idzr + str], val_center, buffers[idzr + str], dz));
		}
	}
	else if (idz == nz - 1) {
		for (int j = 0; j < n_ions; ++j) {
			str = j * n_elements;
			val_center = ions[idfull + str];
			evo_ions[idfull + str] = D_ions[j] * (d2YdX2(ions[idxl + str], val_center, ions[idxr + str], dx) +
				d2YdX2(ions[idyl + str], val_center, ions[idyr + str], dy) +
				d2YdX2(ions[idzl + str], val_center, ions[idzl + str], dz));
		}
		for (int i = 0; i < n_buffers; ++i) {
			str = i * n_elements;
			val_center = buffers[idfull + str];
			evo_buffers[idfull + str] = D_buf[i] * (d2YdX2(buffers[idxl + str], val_center, buffers[idxr + str], dx) +
				d2YdX2(buffers[idyl + str], val_center, buffers[idyr + str], dy) +
				d2YdX2(buffers[idzl + str], val_center, buffers[idzl + str], dz));
		}
	}
	else {
		for (int j = 0; j < n_ions; ++j) {
			str = j * n_elements;
			val_center = ions[idfull + str];
			evo_ions[idfull + str] = D_ions[j] * (d2YdX2(ions[idxl + str], val_center, ions[idxr + str], dx) +
				d2YdX2(ions[idyl + str], val_center, ions[idyr + str], dy) +
				d2YdX2(ions[idzl + str], val_center, ions[idzr + str], dz));
		}
		for (int i = 0; i < n_buffers; ++i) {
			str = i * n_elements;
			val_center = buffers[idfull + str];
			evo_buffers[idfull + str] = D_buf[i] * (d2YdX2(buffers[idxl + str], val_center, buffers[idxr + str], dx) +
				d2YdX2(buffers[idyl + str], val_center, buffers[idyr + str], dy) +
				d2YdX2(buffers[idzl + str], val_center, buffers[idzr + str], dz));
		}
	}
	//reaction
	double R;
	int str2, str3;
	int _idx_buff = 0;
	double R1;
	for (int j = 0; j < n_ions; ++j) {
		str = idfull + j * n_elements;
		R = 0;
		for (int i = 0; i < n_buf_unique; ++i) {
			if (buffer_binds_ion[i + j * n_buf_unique]) {
				str2 = idfull + _idx_buff * n_elements;
				str3 = idfull + i * n_elements;
				R1 = koff_buf[_idx_buff] * buffers[str2] - kon_buf[_idx_buff] * ions[str] * buf_free[str3];
				evo_buffers[str2] += -R1;
				R += R1;
				++_idx_buff;
			}
		}
		evo_ions[str] += R;
	}
}

void DyadRDCPU::_GetFreeBuffers(int idx, int idy, int idz, int idx_batch) {
	int _idx_buff = 0;
	int idfull = _Index(idx, idy, idz, idx_batch);
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

void DyadRDCPU::_UpdateEvo(int idx, int idy, int idz, int idx_batch, double dt) {
	//Separate operator into separate loop
	int idx_SL = 0;
	int idx2;
	int idfull = _Index(idx, idy, idz, idx_batch);
	for (int j = 0; j < n_ions; ++j) {
		idx2 = idfull + j * n_elements;
		evo_ions[idx2] += currents[idx2];
		if (idz == nz - 1 && SL_binds_ion[j]) {
			double s1 = K1[idx_SL] + ions[idx2];
			double s2 = K2[idx_SL] + ions[idx2];
			evo_ions[idx2] *= 1 /  (1 + N1[idx_SL] * K1[idx_SL] / (s1 * s1) + N2[idx_SL] * K2[idx_SL] / (s2 * s2));
			++idx_SL;
		}
		ions[idx2] += evo_ions[idx2] * dt;
	}
	for (int i = 0; i < n_buffers; ++i) {
		idx2 = idfull + i * n_elements;
		buffers[idx2] += evo_buffers[idx2] * dt;
	}
}

//Iteration on calcium and buffers concentrations in dyad 
void DyadRDCPU::RunRD(double dt, int idx_batch) {
	for (int idx = 1; idx < nx - 1; ++idx)
		for (int idy = 1; idy < ny - 1; ++idy)
			for (int idz = 0; idz < nz; ++idz)
				_EvolutionOp(idx, idy, idz, idx_batch,dt);
	for (int idx = 1; idx < nx - 1; ++idx)
		for (int idy = 1; idy < ny - 1; ++idy)
			for (int idz = 0; idz < nz; ++idz) {
				_UpdateEvo(idx, idy, idz, idx_batch, dt);
				_GetFreeBuffers(idx, idy, idz, idx_batch);
			}
}

void DyadRDCPU::RunMC(double dt, int n_thread) {
	channels->RunMC(dt, n_thread, ions_near_channels);
}

double DyadRDCPU::GetL() {
	return R;
}

std::map < std::string, std::vector<double> > DyadRDCPU::GetConcentrations(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions_data.size(); ++i) {
		auto it = std::find(values.begin(), values.end(), ions_data[i]->name);
		if (it != values.end())
			out.insert(std::pair<std::string, std::vector<double> >(ions_data[i]->name, std::vector<double>(ions.begin() + i * n_elements, ions.begin() + (i + 1) * n_elements)));
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
					out.insert(std::pair<std::string, std::vector<double> >(type_name, std::vector<double>(buffers + idx_b * n_elements, buffers + (idx_b + 1) * n_elements)));
				}
			}
		}
	}
	return out;
}

std::map < std::string, std::vector<int> > DyadRDCPU::GetChannelsStates(std::vector<std::string>& values) {
	std::map < std::string, std::vector<int> > out;
	std::string name = "RyR";
	std::vector<int> states;
	for (int i = 0; i < n_threads; ++i)
		for (int j = 0; j < n_channels; ++j)
			states.push_back(channels->GetSRIonChannel(i, j)->GetKineticModel()->GetState());
	out[name] = states;
	return out;
}

std::map <std::string, std::vector<double> > DyadRDCPU::GetIonsNearChannels(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions_data.size(); ++i) {
		auto itb = ions_near_channels.begin() + i * n_threads * n_channels;
		auto ite = ions_near_channels.begin() + (i + 1) * n_threads * n_channels;
		out.insert(std::pair<std::string, std::vector<double> >(ions_data[i]->name, std::vector<double>(itb, ite)));
	}
	return out;
}

std::vector<uint64_t> DyadRDCPU::GetDimensions() {
	return dyad_dims;
}

std::vector<uint64_t> DyadRDCPU::GetChannelsDimensions() {
	return channels_dims;
}
std::vector<uint64_t> DyadRDCPU::GetIonsNearChannelsDimensions() {
	return channels_ions_dims;
}

DyadRDCPU::~DyadRDCPU() {
	delete[] x_idx;
	delete[] y_idx;
	delete[] z_idx;
	delete[] currents;
	delete[] gradients;
	//delete[] ions;
	delete[] evo_ions;
	delete[] evo_buffers;
	delete[] init_ions;
	delete[] buf_free;
	delete[] buffers;
}