#include "CytosolRDMP.h"

#include <algorithm> //find
#include <utility>

__device__ __constant__ float _dr;
__device__ __constant__ float _R;
__device__ __constant__ long _nr;
__device__ __constant__ int _n_threads;
__device__ __constant__ int _n_elements;
__device__ __constant__ int _n_ions;
__device__ __constant__ int _n_ions_dyad;
__device__ __constant__ int _n_buffers;
__device__ __constant__ int _n_buffers_dyad;
__device__ __constant__ double _dt;
__device__ __constant__ int _n_buffers_unique;



cudaStream_t stream_cytoMP;
cudaGraph_t timestep_cyto_graphMP;
cudaGraphExec_t timestep_cyto_instanceMP;

__device__ float d2YdX2_cytoMP(float Yl, float Yc, float Yr, float dX) {
	return (Yr - 2 * Yc + Yl) / (dX * dX);
}

__device__ float dYdX_cytoMP(float Yl, float Yr, float dX) {
	return (Yr - Yl) / (2 * dX);
}

__global__ void set_initials_cytoMP(float* ions, float* init_ions, float* buffers, float* init_buffers, float* buf_free, float* buf_tot, short* ion_buffer_table, double* ions_d, double* buffers_d) {
	int idr_rec = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_sim = threadIdx.y + blockDim.y * blockIdx.y;
	int idfull = idr_rec + _nr * idx_sim;
	int _idx_buff = 0;
	if (idr_rec < _nr && idx_sim < _n_threads) {
		for (int j = 0; j < _n_ions; ++j) {
			int _idx_ion = idfull + j * _n_elements;
			ions[_idx_ion] = init_ions[j];
			ions_d[_idx_ion] = init_ions[j];
			for (int i = 0; i < _n_buffers_unique; ++i) {
				int idx_b = idfull + i * _n_elements;
				if (j == 0) {
					buf_free[idx_b] = buf_tot[i];
				}
				if (ion_buffer_table[i + j * _n_buffers_unique]) {
					float t = init_buffers[_idx_buff];
					buffers[idfull + _idx_buff * _n_elements] = t;
					buffers_d[idfull + _idx_buff * _n_elements] = (double)t;
					buf_free[idx_b] -= t;
					++_idx_buff;
				}
			}
		}
	}
}

__global__ void diffusion_step_cytoMP(float* ions, float* buffers, float* evo_ions, float* evo_buffers, float* D_ions, float* D_buf, short* ion_buffer_table) {
	int idr_rec = threadIdx.x + blockDim.x * blockIdx.x; //2D - (r,n_thread)
	int idx_sim = threadIdx.y + blockDim.y * blockIdx.y;
	int idfull = idr_rec + _nr * idx_sim;
	if (idr_rec < _nr - 1 && idx_sim < _n_threads) {
		if (idr_rec > 0) {
			int idrl = idfull - 1;
			int idrr = idfull + 1;
			float val_center;
			float r = _dr * idr_rec;
			int str;
			for (int j = 0; j < _n_ions; ++j) {
				str = j * _n_elements;
				val_center = ions[idfull + str];
				evo_ions[idfull + str] = D_ions[j] * (2 * dYdX_cytoMP(ions[idrl + str], ions[idrr + str], _dr) / r + d2YdX2_cytoMP(ions[idrl + str], val_center, ions[idrr + str], _dr));
				for (int i = 0; i < _n_buffers; ++i) {
					str = i * _n_elements;
					val_center = buffers[idfull + str];
					evo_buffers[idfull + str] = D_buf[i] * (2 * dYdX_cytoMP(buffers[idrl + str], buffers[idrr + str], _dr) / r + d2YdX2_cytoMP(buffers[idrl + str], val_center, buffers[idrr + str], _dr));
				}
			}
		}
		else {
			int idrr = idfull + 1;
			float val_center;
			int str;
			for (int j = 0; j < _n_ions; ++j) {
				str = j * _n_elements;
				val_center = ions[idfull + str];
				evo_ions[idfull + str] = D_ions[j] * (d2YdX2_cytoMP(ions[idrr + str], val_center, ions[idrr + str], _dr));
				for (int i = 0; i < _n_buffers; ++i) {
					str = i * _n_elements;
					val_center = buffers[idfull + str];
					evo_buffers[idfull + str] = D_buf[i] * (d2YdX2_cytoMP(buffers[idrr + str], val_center, buffers[idrr + str], _dr));
				}
			}
		}
	}
}


__global__ void reaction_step_cytoMP(float* ions, float* buffers, float* evo_ions, float* evo_buffers, float* kon, float* koff, float* buf_free, short* ion_buffer_table, short* is_in_dyad) {
	int idr_rec = threadIdx.x + blockDim.x * blockIdx.x; //2D - (r,n_thread)
	int idx_sim = threadIdx.y + blockDim.y * blockIdx.y;
	int idfull = idr_rec + _nr * idx_sim;
	if (idr_rec < _nr - 1 && idx_sim < _n_threads) {
		float R;
		int str, str2, str3;
		int _idx_buff = 0;
		float R1;
		for (int j = 0; j < _n_ions; ++j) {
			str = idfull + j * _n_elements;
			R = 0;
			for (int i = 0; i < _n_buffers_unique; ++i) {
				if (ion_buffer_table[i + j * _n_buffers_unique]) {
					if (is_in_dyad[_idx_buff] || !(idr_rec * _dr <= _R)) {
						str2 = idfull + _idx_buff * _n_elements;
						str3 = idfull + i * _n_elements;
						R1 = koff[_idx_buff] * buffers[str2] - kon[_idx_buff] * ions[str] * buf_free[str3];
						evo_buffers[str2] = -R1;
						R += R1;
					}
					++_idx_buff;
				}
			}
			evo_ions[str] = R;
		}
	}
}

__global__ void sum_step_cytoMP(float* evo_ions_diffusion, float* ions, float* evo_buffers_diffusion, float* evo_ions_reaction, float* evo_buffers_reaction, float* evo_ions_total, float* evo_buffers_total, float* Jmax, float* K, float* ionsb, short* ion_buffer_table, short* ion_SERCA_table) {
	int idr_rec = threadIdx.x + blockDim.x * blockIdx.x; //2D - (r,n_thread)
	int idx_sim = threadIdx.y + blockDim.y * blockIdx.y;
	int idfull = idr_rec + _nr * idx_sim;
	if (idr_rec < _nr - 1 && idx_sim < _n_threads) {
		int _idx_SERCA = 0;
		int idx2;
		for (int j = 0; j < _n_ions; ++j) {
			idx2 = idfull + j * _n_elements;
			evo_ions_total[idx2] = evo_ions_diffusion[idx2] + evo_ions_reaction[idx2];
			if (ion_SERCA_table[j]) {
				float ions2 = ions[idx2] * ions[idx2];
				float K2 = K[_idx_SERCA] * K[_idx_SERCA];
				float ions2b = ionsb[_idx_SERCA] * ionsb[_idx_SERCA];
				evo_ions_total[idx2] -= Jmax[_idx_SERCA] * (ions2 / (ions2 + K2) - ions2b / (ions2b + K2));
				++_idx_SERCA;
			}
		}
		for (int i = 0; i < _n_buffers; ++i) {
			idx2 = idfull + i * _n_elements;
			evo_buffers_total[idx2] = evo_buffers_diffusion[idx2] + evo_buffers_reaction[idx2];
		}
	}
}

__global__ void evo_step_cytoMP(float* evo_ions_total, double* ions, float* ions_f, float* evo_buffers_total, double* buffers, float* buffers_f, short* ion_buffer_table, double* currents, short* is_in_dyad) {
	int idr_rec = threadIdx.x + blockDim.x * blockIdx.x; //2D - (r,n_thread)
	int idx_sim = threadIdx.y + blockDim.y * blockIdx.y;
	int idfull = idr_rec + _nr * idx_sim;
	int idx_cur = 0;
	if (idr_rec < _nr - 1 && idx_sim < _n_threads) {
		int idx2;
		for (int j = 0; j < _n_ions; ++j) {
			idx2 = idfull + j * _n_elements;
			if (idr_rec * _dr <= _R) {
				evo_ions_total[idx2] += currents[j + idx_sim * (_n_ions_dyad + _n_buffers_dyad)];
			}
			ions[idx2] = ions[idx2] + (double)evo_ions_total[idx2] * _dt;
			ions_f[idx2] = ions[idx2];
		}
		for (int i = 0; i < _n_buffers; ++i) {
			idx2 = idfull + i * _n_elements;
			if (idr_rec * _dr <= _R && is_in_dyad[idx_cur]) {
				evo_buffers_total[idx2] += currents[_n_ions_dyad + idx_cur + idx_sim * (_n_ions_dyad + _n_buffers_dyad)];
				++idx_cur;
			}
			buffers[idx2] = buffers[idx2] + (double)evo_buffers_total[idx2] * _dt;
			buffers_f[idx2] = buffers[idx2];
		}
	}
}

__global__ void get_free_buffers_cytoMP(float* buffers, float* buf_free, float* buf_tot, short* ion_buffer_table) {
	int idr_rec = threadIdx.x + blockDim.x * blockIdx.x; //2D - (r,n_thread)
	int idx_sim = threadIdx.y + blockDim.y * blockIdx.y;
	int idfull = idr_rec + _nr * idx_sim;
	if (idr_rec < _nr - 1 && idx_sim < _n_threads) {
		int _idx_buff = 0;
		for (int j = 0; j < _n_ions; ++j) {
			int idx2 = idfull + j * _n_elements;
			int idx3;
			if (j == 0) {
				for (int i = 0; i < _n_buffers_unique; ++i)
					buf_free[idx2 + i * _n_elements] = buf_tot[i];
			}
			for (int i = 0; i < _n_buffers_unique; ++i) {
				if (ion_buffer_table[i + j * _n_buffers_unique]) {
					idx2 = idfull + _idx_buff * _n_elements;
					idx3 = idfull + i * _n_elements;
					buf_free[idx3] -= buffers[idx2];
					++_idx_buff;
				}
			}
		}
	}
}


__global__ void give_ions_and_buffersMP(double* ions_boundary, double* buffers_boundary, double* ions, double* buffers, short* is_in_dyad) {
	int idfull = threadIdx.x;
	int idx = _R / _dr;
	double alpha = (_R - idx * _dr) / _dr;
	for (int i = 0; i < _n_ions_dyad; ++i) {
		ions_boundary[i + _n_ions_dyad * idfull] = (1 - alpha) * ions[idx + i * _n_elements] + alpha * ions[idx + 1 + i * _n_elements];
	}
	int _idx_buf = 0;
	for (int i = 0; i < _n_buffers; ++i) {
		if (is_in_dyad[i]) {
			buffers_boundary[_idx_buf + _n_buffers_dyad * idfull] = (1 - alpha) * buffers[idx + i * _n_elements] + alpha * buffers[idx + 1 + i * _n_elements];
			++_idx_buf;
		}
	}
}

void CytosolRDMP::Initialize(nlohmann::json& j, nlohmann::json& j_dyad, int nthreads) {
	dr = j["dr"];
	cudaMemcpyToSymbol(_dr, &dr, sizeof(dr));
	r = j["r"];
	block.x = j["CUDA"]["BLOCK X"];
	block.y = j["CUDA"]["BLOCK Y"];
	n_threads = nthreads;
	cudaMemcpyToSymbol(_n_threads, &n_threads, sizeof(n_threads));
	cudaStreamCreate(&(stream_cytoMP));
	nr = r / dr + 1;
	cudaMemcpyToSymbol(_nr, &nr, sizeof(nr));
	cytosol_dims.resize(3);
	cytosol_dims[1] = nr;
	cytosol_dims[2] = n_threads;
	grid.x = (nr + block.x - 1) / block.x, grid.y = (n_threads + block.y - 1) / block.y;
	R_dyad = j_dyad["Radius"];
	cudaMemcpyToSymbol(_R, &R_dyad, sizeof(R_dyad));
	n_elements = nr * n_threads;
	cudaMemcpyToSymbol(_n_elements, &n_elements, sizeof(n_elements));
	int n_ions = 0;
	n_ions_dyad = 0;
	//ions
	for (auto const& el : j_dyad["Ions"].items()) {
		auto pars_ion = el.value();
		ions.push_back(std::make_unique<Ion<float> >(el.key(), pars_ion["D"], pars_ion["Concentration"]));
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
		ions.push_back(std::make_unique<Ion<float> >(el.key(), pars_ion["D"], pars_ion["Concentration"]));
		++n_ions;
		extracell_ions.push_back(pars_ion["Extracell"]);
		ions_bg.push_back(pars_ion["Concentration"]);
		D_ions.push_back(pars_ion["D"]);
	}
	//Buffers
	int idx_buff_dyad = 0;
	for (auto const& el : j_dyad["Buffers"].items()) {
		auto pars_buff = el.value();
		buffers.push_back(std::make_unique<Buffer<float> >(el.key(), pars_buff["Total Concentration"], true));
		total_buf.push_back(pars_buff["Total Concentration"]);
		for (auto const& el_ion : pars_buff["Ions"].items()) {
			auto pars_buff_ion = el_ion.value();
			buffers[buffers.size() - 1]->ions_kinetics[el_ion.key()] = std::make_unique<BufferIon<float> >(pars_buff_ion["D"], pars_buff_ion["kon"], pars_buff_ion["koff"]);
			++idx_buff_dyad;
		}
		int n_ions_buf = buffers[buffers.size() - 1]->ions_kinetics.size();
		std::vector<double> zeroes(n_ions_buf + 1, 0);
		std::vector<std::vector<double> > a(n_ions_buf, zeroes);
		int i_temp = 0;
		for (auto const& i : ions) {
			auto it = buffers[buffers.size() - 1]->ions_kinetics.find(i->name);
			if (it != buffers[buffers.size() - 1]->ions_kinetics.end()) {
				if (j_dyad["Start"] == "Equilibrium") {
					for (int I = 0; I < n_ions_buf; ++I) {
						a[i_temp][I] = it->second->kon * i->Cb;
						if (I == i_temp)
							a[i_temp][I] += it->second->koff;
					}
					a[i_temp][n_ions_buf] = it->second->kon * i->Cb * buffers[buffers.size() - 1]->Ctot;
					++i_temp;
				}
			}
		}
		if (j_dyad["Start"] == "Equilibrium") {
			auto c = GaussElimination(a);
			int t_buf = 0;
			for (auto const& i : ions) {
				auto it = buffers[buffers.size() - 1]->ions_kinetics.find(i->name);
				if (it != buffers[buffers.size() - 1]->ions_kinetics.end()) {
					buffers[buffers.size() - 1]->ions_kinetics[it->first]->initial_C = c[t_buf];
					++t_buf;
				}
			}
		}
	}
	for (auto const& el : j["Buffers"].items()) {
		auto pars_buff = el.value();
		buffers.push_back(std::make_unique<Buffer<float> >(el.key(), pars_buff["Total Concentration"]));
		total_buf.push_back(pars_buff["Total Concentration"]);
		for (auto const& el_ion : pars_buff["Ions"].items()) {
			auto pars_buff_ion = el_ion.value();
			buffers[buffers.size() - 1]->ions_kinetics[el_ion.key()] = std::make_unique<BufferIon<float> >(pars_buff_ion["D"], pars_buff_ion["kon"], pars_buff_ion["koff"]);
		}
		int n_ions_buf = buffers[buffers.size() - 1]->ions_kinetics.size();
		std::vector<double> zeroes(n_ions_buf + 1, 0);
		std::vector<std::vector<double> > a(n_ions_buf, zeroes);
		int i_temp = 0;
		for (auto const& i : ions) {
			auto it = buffers[buffers.size() - 1]->ions_kinetics.find(i->name);
			if (it != buffers[buffers.size() - 1]->ions_kinetics.end()) {
				if (j_dyad["Start"] == "Equilibrium") {
					for (int I = 0; I < n_ions_buf; ++I) {
						a[i_temp][I] = it->second->kon * i->Cb;
						if (I == i_temp)
							a[i_temp][I] += it->second->koff;
					}
					a[i_temp][n_ions_buf] = it->second->kon * i->Cb * buffers[buffers.size() - 1]->Ctot;
					++i_temp;
				}
			}
		}
		if (j_dyad["Start"] == "Equilibrium") {
			auto c = GaussElimination(a);
			int t_buf = 0;
			for (auto const& i : ions) {
				auto it = buffers[buffers.size() - 1]->ions_kinetics.find(i->name);
				if (it != buffers[buffers.size() - 1]->ions_kinetics.end()) {
					buffers[buffers.size() - 1]->ions_kinetics[it->first]->initial_C = c[t_buf];
					++t_buf;
				}
			}
		}
	}
	int n_buf_unique = buffers.size();
	cudaMemcpyToSymbol(_n_buffers_unique, &n_buf_unique, sizeof(n_buffers));
	int idx_buf = 0;
	for (auto const& i : ions) {
		for (auto const& b : buffers) {
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
	cudaMalloc(&d_is_in_dyad, is_in_dyad.size() * sizeof(short));
	cudaMemcpy(d_is_in_dyad, is_in_dyad.data(), is_in_dyad.size() * sizeof(short), cudaMemcpyHostToDevice);
	for (auto const& el : j["SERCA"].items()) {
		auto pars_SR = el.value();
		ions_and_SR[el.key()] = std::make_unique<IonSERCA<float> >(el.key(), pars_SR["Jmax"], pars_SR["Kup"]);
	}
	for (auto const& i : ions) {
		auto it = ions_and_SR.find(i->name);
		if (it != ions_and_SR.end()) {
			SR_uptakes_ion.push_back(1);
			Jmax.push_back(it->second->Jmax);
			Kup.push_back(it->second->Kup);
		}
		else
			SR_uptakes_ion.push_back(0);
	}
	dt_is_set = false;
	n_blocks_init = (n_elements + threads_per_block - 1) / threads_per_block;
	cudaMalloc(&d_buffer_binds_ion, buffer_binds_ion.size() * sizeof(short));
	cudaMemcpy(d_buffer_binds_ion, buffer_binds_ion.data(), buffer_binds_ion.size() * sizeof(short), cudaMemcpyHostToDevice);
	cudaMalloc(&d_SR_uptakes_ion, SR_uptakes_ion.size() * sizeof(short));
	cudaMemcpy(d_SR_uptakes_ion, SR_uptakes_ion.data(), SR_uptakes_ion.size() * sizeof(short), cudaMemcpyHostToDevice);
	cudaMalloc(&d_ionsb, ions_bg.size() * sizeof(float));
	cudaMemcpy(d_ionsb, ions_bg.data(), ions_bg.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_ions_boundary, n_ions * n_threads * sizeof(double));
	cudaMalloc(&d_ions, n_elements * n_ions * sizeof(double));
	cudaMalloc(&d_ions_f, n_elements * n_ions * sizeof(float));
	cudaMalloc(&d_currents, (n_ions_dyad + idx_buff_dyad) * n_threads * sizeof(double));
	zero_vector.resize((n_ions_dyad + idx_buff_dyad) * n_threads);
	std::fill(zero_vector.begin(), zero_vector.end(), 0);
	_SetNumIonsandBuffsinDyad(n_ions_dyad, idx_buff_dyad);
	cudaMalloc(&d_init_ions, n_ions * sizeof(float));
	std::vector<float> ions_b;
	for (auto const& i : ions)
		ions_b.push_back(i->Cb);
	cudaMemcpyToSymbol(_n_buffers, &idx_buf, sizeof(idx_buf));
	cudaMemcpy(d_init_ions, ions_b.data(), n_ions * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_buffers_boundary, idx_buf * n_threads * sizeof(double));
	cudaMalloc(&d_buffers, n_elements * idx_buf * sizeof(double));
	cudaMalloc(&evo_ions_diffusion, n_elements * n_ions * sizeof(float));
	cudaMalloc(&evo_buffers_diffusion, n_elements * idx_buf * sizeof(float));
	cudaMalloc(&evo_ions_reaction, n_elements * n_ions * sizeof(float));
	cudaMalloc(&evo_buffers_reaction, n_elements * idx_buf * sizeof(float));
	cudaMalloc(&evo_ions_total, n_elements * n_ions * sizeof(float));
	cudaMalloc(&evo_buffers_total, n_elements * idx_buf * sizeof(float));
	cudaMalloc(&d_buffers_f, n_elements * idx_buf * sizeof(float));
	cudaMalloc(&d_buffers_free, n_elements * idx_buf * sizeof(float));
	cudaMalloc(&d_D_ions, D_ions.size() * sizeof(float));
	cudaMemcpy(d_D_ions, D_ions.data(), D_ions.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_D_buf, D_buf.size() * sizeof(float));
	cudaMemcpy(d_D_buf, D_buf.data(), D_buf.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_kon, kon_buf.size() * sizeof(float));
	cudaMemcpy(d_kon, kon_buf.data(), kon_buf.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_koff, koff_buf.size() * sizeof(float));
	cudaMemcpy(d_koff, koff_buf.data(), koff_buf.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_CTot, total_buf.size() * sizeof(float));
	cudaMemcpy(d_CTot, total_buf.data(), total_buf.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_buffer_binds_ion, buffer_binds_ion.size() * sizeof(short));
	cudaMemcpy(d_buffer_binds_ion, buffer_binds_ion.data(), buffer_binds_ion.size() * sizeof(short), cudaMemcpyHostToDevice);
	cudaMalloc(&d_init_buffers, idx_buf * sizeof(float));
	cudaMemcpy(d_init_buffers, buf_init.data(), idx_buf * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_Jmax, Jmax.size() * sizeof(float));
	cudaMemcpy(d_Jmax, Jmax.data(), Jmax.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_Kup, Kup.size() * sizeof(float));
	cudaMemcpy(d_Kup, Kup.data(), Kup.size() * sizeof(float), cudaMemcpyHostToDevice);
}

CytosolRDMP::CytosolRDMP(nlohmann::json& j, nlohmann::json& j_dyad, int nthreads) {
	Initialize(j, j_dyad, nthreads);
}

std::vector<double> CytosolRDMP::GaussElimination(std::vector<std::vector<double> >& a) {
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

CytosolRDMP::CytosolRDMP(nlohmann::json& j, nlohmann::json& j_dyad, nlohmann::json& j_nsr, int nthreads) {
	for (auto const& el : j_nsr["Ions"].items()) {
		auto pars_ion = el.value();
		nsr_ions.push_back(pars_ion["Concentration (uM)"]);
	}
	Initialize(j, j_dyad, nthreads);
}

void CytosolRDMP::_SetNumIonsandBuffsinDyad(int nions, int nbuf) {
	n_ions_dyad = nions;
	cudaMemcpyToSymbol(_n_ions_dyad, &n_ions_dyad, sizeof(n_ions_dyad));
	cudaMemcpyToSymbol(_n_ions, &nions, sizeof(n_ions_dyad));
	n_buffs_dyad = nbuf;
	cudaMemcpyToSymbol(_n_buffers_dyad, &n_buffs_dyad, sizeof(n_buffs_dyad));
}

std::vector<double>& CytosolRDMP::GetExtraCellularIons() {
	return extracell_ions;
}


void CytosolRDMP::_SetupStepGraph() {
	cudaStreamBeginCapture(stream_cytoMP, cudaStreamCaptureModeGlobal);
	diffusion_step_cytoMP << <grid, threads_per_block, 0, stream_cytoMP >> > (d_ions_f, d_buffers_f, evo_ions_diffusion, evo_buffers_diffusion, d_D_ions, d_D_buf, d_buffer_binds_ion);
	reaction_step_cytoMP << <grid, threads_per_block, 0, stream_cytoMP >> > (d_ions_f, d_buffers_f, evo_ions_reaction, evo_buffers_reaction, d_kon, d_koff, d_buffers_free, d_buffer_binds_ion, d_is_in_dyad);
	sum_step_cytoMP << <grid, threads_per_block, 0, stream_cytoMP >> > (evo_ions_diffusion, d_ions_f, evo_buffers_diffusion, evo_ions_reaction, evo_buffers_reaction, evo_ions_total, evo_buffers_total, d_Jmax, d_Kup, d_ionsb, d_buffer_binds_ion, d_SR_uptakes_ion);
	evo_step_cytoMP << <grid, threads_per_block, 0, stream_cytoMP >> > (evo_ions_total, d_ions, d_ions_f, evo_buffers_total, d_buffers, d_buffers_f, d_buffer_binds_ion, d_currents, d_is_in_dyad); //DOUBLE PRECISION KERNEL 
	get_free_buffers_cytoMP << <grid, threads_per_block, 0, stream_cytoMP >> > (d_buffers_f, d_buffers_free, d_CTot, d_buffer_binds_ion);
	cudaStreamEndCapture(stream_cytoMP, &timestep_cyto_graphMP);
	cudaGraphInstantiate(&timestep_cyto_instanceMP, timestep_cyto_graphMP, NULL, NULL, 0);
}


void CytosolRDMP::Update(double*& dyad_currents, const std::vector<double>& nsr_ions) {
	if (dyad_currents != d_currents) {
		d_currents = dyad_currents;
		_SetupStepGraph();
	}
}

//Iteration on calcium and buffers concentrations in the cytosol 
void CytosolRDMP::RunRD(double dt,int idx_b) {
	if (!dt_is_set) {
		cudaMemcpyToSymbol(_dt, &dt, sizeof(double));
		dt_is_set = true;
	}
	cudaGraphLaunch(timestep_cyto_instanceMP, stream_cytoMP);
}

void CytosolRDMP::GetIonsandBuffers(double*& d_ions_boundary_out, double*& d_buffers_boundary_out) {
	give_ions_and_buffersMP << <1, n_threads >> > (d_ions_boundary, d_buffers_boundary, d_ions, d_buffers, d_is_in_dyad);
	d_ions_boundary_out = d_ions_boundary;
	d_buffers_boundary_out = d_buffers_boundary;
}


bool CytosolRDMP::UsesGPU() {
	return true;
}

std::map < std::string, std::vector<double> > CytosolRDMP::GetConcentrations(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions.size(); ++i) {
		auto it = std::find(values.begin(), values.end(), ions[i]->name);
		if (it != values.end()) {
			out.insert(std::pair<std::string, std::vector<double> >(ions[i]->name, std::vector<double>(n_elements)));
			cudaMemcpyAsync(out[ions[i]->name].data(), d_ions + i * n_elements, n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream_cytoMP);
		}
	}
	for (int i = 0; i < buffers.size(); ++i) {
		auto it = std::find(values.begin(), values.end(), buffers[i]->name);
		if (it != values.end()) {
			auto buf_name = buffers[i]->name;
			for (auto const& ion : ions) {
				auto it2 = buffers[i]->ions_kinetics.find(ion->name);
				if (it2 != buffers[i]->ions_kinetics.end()) {
					auto ion_name = ion->name;
					auto type_name = ion_name + std::string("-") + buf_name;
					out.insert(std::pair<std::string, std::vector<double> >(type_name, std::vector<double>(n_elements)));
					int idx_b = idx_bufion[buf_name][ion_name];
					cudaMemcpyAsync(out[type_name].data(), d_buffers + idx_b * n_elements, n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream_cytoMP);
				}
			}
		}
	}
	cudaDeviceSynchronize();
	return out;
}

void CytosolRDMP::Reset() {
	set_initials_cytoMP << <grid, block >> > (d_ions_f, d_init_ions, d_buffers_f, d_init_buffers, d_buffers_free, d_CTot, d_buffer_binds_ion, d_ions, d_buffers);
	cudaMemcpy(d_currents, zero_vector.data(), zero_vector.size() * sizeof(double), cudaMemcpyHostToDevice);
}

std::vector<uint64_t> CytosolRDMP::GetDimensions() {
	return cytosol_dims;
}

CytosolRDMP::~CytosolRDMP() {
	cudaFree(&d_buffer_binds_ion);
	cudaFree(&d_SR_uptakes_ion);
	cudaFree(&d_ionsb);
	cudaFree(&d_ions_boundary);
	cudaFree(&d_ions);
	cudaFree(&d_ions_f);
	cudaFree(&d_currents);
	cudaFree(&d_init_ions);
	cudaFree(&d_buffers_boundary);
	cudaFree(&d_buffers);
	cudaFree(&evo_ions_total);
	cudaFree(&evo_buffers_total);
	cudaFree(&d_buffers_f);
	cudaFree(&d_buffers_free);
	cudaFree(&d_D_buf);
	cudaFree(&d_kon);
	cudaFree(&d_koff);
	cudaFree(&d_CTot);
	cudaFree(&d_buffer_binds_ion);
	cudaFree(&d_init_buffers);
	cudaFree(&d_Jmax);
	cudaFree(&d_Kup);
	cudaFree(&evo_ions_diffusion);
	cudaFree(&evo_buffers_diffusion);
	cudaFree(&evo_ions_reaction);
	cudaFree(&evo_buffers_reaction);
	cudaGraphExecDestroy(timestep_cyto_instanceMP);
	cudaGraphDestroy(timestep_cyto_graphMP);
	cudaStreamDestroy(stream_cytoMP);
}