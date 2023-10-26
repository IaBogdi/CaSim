#include "DyadRD2DwCytosol.h"

#include <algorithm> //find
#include <utility>
#include <iostream>

const int MAX_IONS = 10;
const int MAX_UNIQUE_BUFFERS = 30;
const int MAX_BUFFERS = 200;
const int MAX_CHANNELS = 100;
__device__ __constant__ float _dxd; //dyad
__device__ __constant__ float _dxc; //cyto
__device__ __constant__ float _dyd; //dyad
__device__ __constant__ float _dyc; //cyto
__device__ __constant__ int _n_threads;
__device__ __constant__ int  _nx;
__device__ __constant__ int  _ny;
__device__ __constant__ int _nx_dyad_start;
__device__ __constant__ int _nx_dyad_end;
__device__ __constant__ int _ny_dyad_start;
__device__ __constant__ int _ny_dyad_end;
__device__ __constant__ int  _n_elements;
__device__ __constant__ int  _n_elements_per_thread;
__device__ __constant__ int _n_channels;
__device__ __constant__ int _n_ions;
__device__ __constant__ int _n_buffers; //number of buffers with diffusion
__device__ __constant__ int _n_buffers_unique; //total number of buffers
__device__ __constant__ double _dt;
__device__ __constant__ short _ion_buffer_table[MAX_BUFFERS];
__device__ __constant__ int _channel_x[MAX_CHANNELS];
__device__ __constant__ int _channel_y[MAX_CHANNELS];
__device__ __constant__ float _D_ions[MAX_IONS];
__device__ __constant__ float _D_buf[MAX_BUFFERS];
__device__ __constant__ float _kon[MAX_BUFFERS];
__device__ __constant__ float _koff[MAX_BUFFERS];
__device__ __constant__ float _buf_tot[MAX_BUFFERS];
__device__ __constant__ short _ion_SL_table[MAX_IONS];
__device__ __constant__ short _is_in_dyad[MAX_UNIQUE_BUFFERS];
__device__ __constant__ float _Jmax[MAX_IONS];
__device__ __constant__ float _Kserca[MAX_IONS];
__device__ __constant__ float _ionsb[MAX_IONS];
__device__ __constant__ float _K1[MAX_IONS];
__device__ __constant__ float _K2[MAX_IONS];
__device__ __constant__ float _N1[MAX_IONS];
__device__ __constant__ float _N2[MAX_IONS];
__device__ __constant__ short _ion_SERCA_table[MAX_IONS];


cudaStream_t stream_dyad_cyto_2d;
cudaGraph_t timestep_graph_cyto_2d, update_graph_cyto_2d;
cudaGraphExec_t timestep_instance_cyto_2d, update_instance_cyto_2d;

__device__ __forceinline__ float d2YdX2MP2DwCyto(float Yl, float Yc, float Yr, float dXl, float dXr) {
	return (Yr * dXl + Yl * dXr - Yc * (dXl + dXr)) / (0.5 * dXl * dXr * (dXr + dXl));
}

__device__ __forceinline__ int IndexMP2DwCyto(int idx, int idy, int idw) {
	return idw + _n_threads * (idy + _ny * idx);
}

//initial values (Ca + buffers)
__global__ void set_initialsMP2DwCyto(float* ions, float* buffers, float* init_buffers, float* currents, float* buf_free, double* ions_d, double* buffers_d) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_batch = threadIdx.z + blockDim.z * blockIdx.z;
	int idfull = IndexMP2DwCyto(idx, idy, idx_batch);
	if (idx < _nx && idy < _ny && idx_batch < _n_threads) {
		int _idx_buff = 0;
		for (int j = 0; j < _n_ions; ++j) {
			int _idx_ion = idfull + j * _n_elements;
			ions[_idx_ion] = _ionsb[j];
			ions_d[_idx_ion] = (double)_ionsb[j];
			currents[_idx_ion] = 0;
			for (int i = 0; i < _n_buffers_unique; ++i) {
				int idx_b = idfull + i * _n_elements;
				if (j == 0) {
					buf_free[idx_b] = _buf_tot[i];
				}
				if (_ion_buffer_table[i + j * _n_buffers_unique]) {
					double t = init_buffers[_idx_buff];
					buffers[idfull + _idx_buff * _n_elements] = t;
					buffers_d[idfull + _idx_buff * _n_elements] = (double)t;
					buf_free[idx_b] -= t;
					++_idx_buff;
				}
			}
		}
	}
}
// set currents values based on input float array and channels positions

__global__ void set_currents2DCyto(float* current_grid, float* current_values) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n_total_channels = _n_threads * _n_channels;
	int idx_grid = IndexMP2DwCyto(_channel_x[threadIdx.x],_channel_y[threadIdx.x],blockIdx.x);
	for (int j = 0; j < _n_ions; ++j)
		current_grid[idx_grid + j * _n_elements] = current_values[idx + j * n_total_channels];
}

__global__ void get_ions_near_channels2DwCytosol(double* ions, double* ions_near_channels) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n_total_channels = _n_threads * _n_channels;
	int idx_grid = IndexMP2DwCyto(_channel_x[threadIdx.x], _channel_y[threadIdx.x], blockIdx.x);
	for (int j = 0; j < _n_ions; ++j) {
		ions_near_channels[idx + j * n_total_channels] = ions[idx_grid + j * _n_elements];
	}
}

__device__ __forceinline__ void loadSharedData(float* source, float* sharedDest, int sourceIndex, int sharedIndex, int length) {
	for (int i = 0; i < length; ++i) {
		sharedDest[sharedIndex + i * (blockDim.x + 2) * (blockDim.y + 2) * blockDim.z] = source[sourceIndex + i * _n_elements];
	}
}

__device__ __forceinline__ bool is_within_dyad(int x, int y) {
	return (x <= _nx_dyad_end) && (x >= _nx_dyad_start) && (y <= _ny_dyad_end) && (y >= _ny_dyad_start);
}

__global__ void evolution_operator(float* ions, float* buffers, float* evo_ions, float* evo_buffers, float* buf_free, float* currents) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_batch = threadIdx.z + blockDim.z * blockIdx.z;
	//first, create shared memory arrays
	extern __shared__ float sharedData[];
	float* ions_sh = sharedData;
	float* buffers_sh = &sharedData[_n_ions * (blockDim.x + 2) * (blockDim.y + 2) * blockDim.z];
	if (idx > 0 && idx < _nx - 1 && idy > 0 && idy < _ny - 1 && idx_batch < _n_threads) { //Dirichlet BC
		int n_elem_sh = (blockDim.x + 2) * (blockDim.y + 2) * blockDim.z;
		int idfull = IndexMP2DwCyto(idx, idy, idx_batch);
		int idfull_sh = threadIdx.z + blockDim.z * ((threadIdx.y + 1) + (blockDim.y + 2) * (threadIdx.x + 1));
		loadSharedData(ions, ions_sh, idfull, idfull_sh, _n_ions);
		loadSharedData(buffers, buffers_sh, idfull, idfull_sh, _n_buffers);
		// Handle boundary cells
		int idfull_temp, idfull_sh_temp;
		if (threadIdx.x == 0 || idx == 1) {
			idfull_temp = IndexMP2DwCyto(idx - 1, idy, idx_batch);
			int idx_sh = idx == 1 ? threadIdx.x : 0;
			idfull_sh_temp = threadIdx.z + blockDim.z * ((threadIdx.y + 1) + (blockDim.y + 2) * idx_sh);
			loadSharedData(ions, ions_sh, idfull_temp, idfull_sh_temp, _n_ions);
			loadSharedData(buffers, buffers_sh, idfull_temp, idfull_sh_temp, _n_buffers);
		}
		if (threadIdx.x == blockDim.x - 1 || idx == _nx - 2) {
			idfull_temp = IndexMP2DwCyto(idx + 1, idy, idx_batch);
			int idx_sh = idx == _nx - 2 ? threadIdx.x + 2 : blockDim.x + 1;
			idfull_sh_temp = threadIdx.z + blockDim.z * ((threadIdx.y + 1) + (blockDim.y + 2) * idx_sh);
			loadSharedData(ions, ions_sh, idfull_temp, idfull_sh_temp, _n_ions);
			loadSharedData(buffers, buffers_sh, idfull_temp, idfull_sh_temp, _n_buffers);
		}
		if (threadIdx.y == 0 || idy == 1) {
			idfull_temp = IndexMP2DwCyto(idx, idy - 1, idx_batch);
			int idy_sh = idy == 1 ? threadIdx.y : 0;
			idfull_sh_temp = threadIdx.z + blockDim.z * (idy_sh + (blockDim.y + 2) * (threadIdx.x + 1));
			loadSharedData(ions, ions_sh, idfull_temp, idfull_sh_temp, _n_ions);
			loadSharedData(buffers, buffers_sh, idfull_temp, idfull_sh_temp, _n_buffers);
		}
		if (threadIdx.y == blockDim.y - 1 || idy == _ny - 2) {
			idfull_temp = IndexMP2DwCyto(idx, idy + 1, idx_batch);
			int idy_sh = idy == _ny - 2 ? threadIdx.y + 2 : blockDim.y + 1;
			idfull_sh_temp = threadIdx.z + blockDim.z * (idy_sh + (blockDim.y + 2) * (threadIdx.x + 1));
			loadSharedData(ions, ions_sh, idfull_temp, idfull_sh_temp, _n_ions);
			loadSharedData(buffers, buffers_sh, idfull_temp, idfull_sh_temp, _n_buffers);
		}
		__syncthreads();
		//diffusion part
		int idxl = threadIdx.z + blockDim.z * ((threadIdx.y + 1) + (blockDim.y + 2) * threadIdx.x);
		int idxr = threadIdx.z + blockDim.z * ((threadIdx.y + 1) + (blockDim.y + 2) * (threadIdx.x + 2));
		int idyl = threadIdx.z + blockDim.z * (threadIdx.y + (blockDim.y + 2) * (threadIdx.x + 1));
		int idyr = threadIdx.z + blockDim.z * ((threadIdx.y + 2) + (blockDim.y + 2) * (threadIdx.x + 1));
		bool center_within_dyad = is_within_dyad(idx, idy);
		float dxl = center_within_dyad && is_within_dyad(idx - 1, idy) ? _dxd : _dxc;
		float dxr = center_within_dyad && is_within_dyad(idx + 1, idy) ? _dxd : _dxc;
		float dyl = center_within_dyad && is_within_dyad(idx, idy - 1) ? _dyd : _dyc;
		float dyr = center_within_dyad && is_within_dyad(idx, idy + 1) ? _dyd : _dyc;
		int str, str_sh;
		for (int j = 0; j < _n_ions; ++j) {
			str_sh = j * n_elem_sh;
			str = j * _n_elements;
			evo_ions[idfull + str] = _D_ions[j] * (d2YdX2MP2DwCyto(ions_sh[idxl + str_sh], ions_sh[idfull_sh + str_sh], ions_sh[idxr + str_sh],dxl,dxr) +
				d2YdX2MP2DwCyto(ions_sh[idyl + str_sh], ions_sh[idfull_sh + str_sh], ions_sh[idyr + str_sh], dyl,dyr));
		}
		for (int j = 0; j < _n_buffers; ++j) {
			str_sh = j * n_elem_sh;
			str = j * _n_elements;
			evo_buffers[idfull + str] = _D_buf[j] * (d2YdX2MP2DwCyto(buffers_sh[idxl + str_sh], buffers_sh[idfull_sh + str_sh], buffers_sh[idxr + str_sh], dxl, dxr) +
				d2YdX2MP2DwCyto(buffers_sh[idyl + str_sh], buffers_sh[idfull_sh + str_sh], buffers_sh[idyr + str_sh], dyl, dyr));
		}
		//reaction part (non-dyadic buffer (troponin) is taken into account)
		float R;
		int str2, str2_sh, str3;
		int _idx_buff = 0;
		float R1;
		for (int j = 0; j < _n_ions; ++j) {
			str_sh = idfull_sh + j * n_elem_sh;
			str = idfull + j * _n_elements;
			R = 0;
			for (int i = 0; i < _n_buffers_unique; ++i) {
				if (_ion_buffer_table[i + j * _n_buffers_unique]) {
					if (_is_in_dyad[_idx_buff] || !is_within_dyad(idx,idy)) {
						str2_sh = idfull_sh + _idx_buff * n_elem_sh;
						str2 = idfull + _idx_buff * _n_elements;
						str3 = idfull + i * _n_elements;
						R1 = _koff[_idx_buff] * buffers_sh[str2_sh] - _kon[_idx_buff] * ions_sh[str_sh] * buf_free[str3];
						evo_buffers[str2] += -R1;
						R += R1;
					}
					++_idx_buff;
				}
			}
			evo_ions[str] += R;
		}
		//current part (dyad - current + SL, cytosol - SERCA)
		if (is_within_dyad(idx, idy)) { //dyadic domain
			int _idx_SL = 0;
			for (int j = 0; j < _n_ions; ++j) {
				int idx2sh = idfull_sh + j * n_elem_sh;
				int idx2 = idfull + j * _n_elements;
				evo_ions[idx2] += currents[idx2];
				if (_ion_SL_table[j]) {
					float s1 = _K1[_idx_SL] + ions_sh[idx2sh];
					float s2 = _K2[_idx_SL] + ions_sh[idx2sh];
					evo_ions[idx2] *= 1 / (1 + _N1[_idx_SL] * _K1[_idx_SL] / (s1 * s1) + _N2[_idx_SL] * _K2[_idx_SL] / (s2 * s2));
					++_idx_SL;
				}
			}
		}
		else { //cytosolic domain
			int _idx_SERCA = 0;
			for (int j = 0; j < _n_ions; ++j) {
				int idx2sh = idfull_sh + j * n_elem_sh;
				int idx2 = idfull + j * _n_elements;
				if (_ion_SERCA_table[j]) {
					float ions2 = ions_sh[idx2sh] * ions_sh[idx2sh];
					float K2 = _Kserca[_idx_SERCA] * _Kserca[_idx_SERCA];
					float ions2b = _ionsb[_idx_SERCA] * _ionsb[_idx_SERCA];
					evo_ions[idx2] -= _Jmax[_idx_SERCA] * (ions2 / (ions2 + K2) - ions2b / (ions2b + K2));
					++_idx_SERCA;
				}
			}
		}
	}
}

__global__ void evo_step2DCyto(float* evo_ions_total, double* ions, float* ions_f, float* evo_buffers_total, double* buffers, float* buffers_f) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_batch = threadIdx.z + blockDim.z * blockIdx.z;
	int idfull = IndexMP2DwCyto(idx, idy, idx_batch);
	double tot_double;
	if (idx > 0 && idx < _nx - 1 && idy > 0 && idy < _ny - 1 && idx_batch < _n_threads) {
		int idx2;
		for (int j = 0; j < _n_ions; ++j) {
			idx2 = idfull + j * _n_elements;
			tot_double = (double)evo_ions_total[idx2];
			ions[idx2] += tot_double * _dt;
			ions_f[idx2] = ions[idx2];
		}
		for (int i = 0; i < _n_buffers; ++i) {
			idx2 = idfull + i * _n_elements;
			tot_double = (double)evo_buffers_total[idx2];
			buffers[idx2] += +tot_double * _dt;
			buffers_f[idx2] = buffers[idx2];
		}
	}
}



__global__ void get_free_buffers2DCyto(float* buffers, float* buf_free) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_batch = threadIdx.z + blockDim.z * blockIdx.z;
	int idfull = IndexMP2DwCyto(idx, idy, idx_batch);
	if (idx < _nx && idy < _ny && idx_batch < _n_threads) {
		if (!((idx == 0 || idx == _nx - 1) || (idy == 0 || idy == _ny - 1))) {
			int _idx_buff = 0;
			for (int j = 0; j < _n_ions; ++j) {
				int idx2 = idfull + j * _n_elements;
				int idx3;
				if (j == 0) {
					for (int i = 0; i < _n_buffers_unique; ++i)
						buf_free[idx2 + i * _n_elements] = _buf_tot[i];
				}
				for (int i = 0; i < _n_buffers_unique; ++i) {
					if (_ion_buffer_table[i + j * _n_buffers_unique]) {
						idx2 = idfull + _idx_buff * _n_elements;
						idx3 = idfull + i * _n_elements;
						buf_free[idx3] -= buffers[idx2];
						++_idx_buff;
					}
				}
			}
		}
	}
}


DyadRD2DwCytosol::DyadRD2DwCytosol(nlohmann::json& j, nlohmann::json& j_cyto, int nthreads) {
	channels = std::make_unique<DyadChannels>(j, nthreads);
	float dxd = j["dx"];
	float dxc = j_cyto["dx"];
	cudaMemcpyToSymbol(_dxd, &dxd, sizeof(float));
	cudaMemcpyToSymbol(_dxc, &dxc, sizeof(float));
	float dyd = j["dy"];
	float dyc = j_cyto["dy"];
	cudaMemcpyToSymbol(_dyd, &dyd, sizeof(float));
	cudaMemcpyToSymbol(_dyc, &dyc, sizeof(float));
	x = j_cyto["x"];
	y = j_cyto["y"];
	z = j["z"];
	float x_dyad = j["x"];
	float y_dyad = j["y"];
	blockx = j["CUDA"]["BLOCK X"];
	blocky = j["CUDA"]["BLOCK Y"];
	blockz = j["CUDA"]["BLOCK Z"];
	block.x = blockx, block.y = blocky, block.z = blockz;
	n_threads = nthreads;
	cudaStreamCreate(&(stream_dyad_cyto_2d));
	cudaMemcpyToSymbol(_n_threads, &n_threads, sizeof(n_threads));
	int nx_dyad_start = (x - x_dyad) * 0.5 / dxc;
	int ny_dyad_start = (y - y_dyad) * 0.5 / dyc;
	cudaMemcpyToSymbol(_nx_dyad_start, &nx_dyad_start, sizeof(nx_dyad_start));
	cudaMemcpyToSymbol(_ny_dyad_start, &ny_dyad_start, sizeof(ny_dyad_start));
	int nx_dyad_end = nx_dyad_start + x_dyad / dxd;
	int ny_dyad_end = ny_dyad_start + y_dyad / dyd;
	cudaMemcpyToSymbol(_nx_dyad_end, &nx_dyad_end, sizeof(nx_dyad_end));
	cudaMemcpyToSymbol(_ny_dyad_end, &ny_dyad_end, sizeof(ny_dyad_end));
	nx = nx_dyad_end + nx_dyad_start + 1;
	cudaMemcpyToSymbol(_nx, &nx, sizeof(nx));
	ny = ny_dyad_end + ny_dyad_start + 1;
	cudaMemcpyToSymbol(_ny, &ny, sizeof(ny));
	dyad_dims.resize(4);
	dyad_dims[1] = nx;
	dyad_dims[2] = ny;
	dyad_dims[3] = n_threads;
	int n_elements_thread = nx * ny;
	cudaMemcpyToSymbol(_n_elements_per_thread, &n_elements_thread, sizeof(n_elements_thread));
	grid.x = (nx + block.x - 1) / blockx, grid.y = (ny + block.y - 1) / blocky, grid.z = (n_threads + block.z - 1) / blockz;
	n_elements = nx * ny * n_threads;
	cudaMemcpyToSymbol(_n_elements, &n_elements, sizeof(n_elements));
	threads_per_block = block.x * block.y * block.z;
	//Ions
	n_ions = 0;
	int n_ions_dyad = 0;
	//ions
	for (auto const& el : j["Ions"].items()) {
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
	for (auto const& el : j_cyto["Ions"].items()) {
		auto pars_ion = el.value();
		ions.push_back(std::make_unique<Ion<float> >(el.key(), pars_ion["D"], pars_ion["Concentration"]));
		++n_ions;
		extracell_ions.push_back(pars_ion["Extracell"]);
		ions_bg.push_back(pars_ion["Concentration"]);
		D_ions.push_back(pars_ion["D"]);
	}
	total_sr_current.resize(n_ions * n_threads);
	//Channels
	sl_size = channels->GetNSLIonChannels();
	sr_size = channels->GetNSRIonChannels();
	n_channels = sl_size + sr_size;
	for (int j = 0; j < sr_size; ++j) {
		auto p = channels->GetSRIonChannel(0, j)->GetCoordinates()->GetCoordsOnGrid(dxd, dyd, dz);
		x_idx_v.push_back(p[0] + nx_dyad_start);
		y_idx_v.push_back(p[1] + ny_dyad_start);
	}
	for (int j = sr_size; j < n_channels; ++j) {
		auto p = channels->GetSLIonChannel(0, j - sr_size)->GetCoordinates()->GetCoordsOnGrid(dxd, dyd, dz);
		x_idx_v.push_back(p[0] + nx_dyad_start);
		y_idx_v.push_back(p[1] + ny_dyad_start);
	}
	cudaMemcpyToSymbol(_n_channels, &n_channels, sizeof(n_channels));
	n_elements_near_channels = n_ions * n_channels * n_threads;
	currents.resize(n_elements_near_channels);
	channels_ions_dims.resize(3);
	channels_ions_dims[1] = n_channels;
	channels_ions_dims[2] = n_threads;
	channels_dims.resize(3);
	channels_dims[1] = n_channels;
	channels_dims[2] = n_threads;
	cudaMalloc(&d_currents, n_channels * n_ions * n_threads * sizeof(float));
	cudaMemcpyToSymbol(_channel_x, x_idx_v.data(), sizeof(int) * x_idx_v.size());
	cudaMemcpyToSymbol(_channel_y, y_idx_v.data(), sizeof(int) * y_idx_v.size());
	ions_near_channels.resize(n_elements_near_channels);
	cudaMemcpyToSymbol(_n_ions, &n_ions, sizeof(n_ions));
	//Buffers
	int idx_buff_dyad = 0;
	for (auto const& el : j["Buffers"].items()) {
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
				if (j["Start"] == "Equilibrium") {
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
		if (j["Start"] == "Equilibrium") {
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
	for (auto const& el : j_cyto["Buffers"].items()) {
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
				if (j["Start"] == "Equilibrium") {
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
		if (j["Start"] == "Equilibrium") {
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
	//loop ions and buffers so that the GPU computation would go smoothly
	idx_buf = 0;
	for (auto const& i : ions)
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
	cudaMemcpyToSymbol(_is_in_dyad, is_in_dyad.data(), sizeof(int) * is_in_dyad.size());
	for (auto const& el : j["Sarcolemma"].items()) {
		auto pars_SL = el.value();
		ions_and_SL[el.key()] = std::make_unique<IonSarcolemma<float> >(pars_SL["N1"], pars_SL["K1"], pars_SL["N2"], pars_SL["K2"]);
	}
	for (auto const& el : j["SERCA"].items()) {
		auto pars_SR = el.value();
		ions_and_SR[el.key()] = std::make_unique<IonSERCA<float> >(el.key(), pars_SR["Jmax"], pars_SR["Kup"]);
	}
	for (auto const& i : ions) {
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
	cudaMemcpyToSymbol(_K1, K1.data(), sizeof(float) * K1.size());
	cudaMemcpyToSymbol(_K2, K2.data(), sizeof(float) * K2.size());
	cudaMemcpyToSymbol(_N1, N1.data(), sizeof(float)* N1.size());
	cudaMemcpyToSymbol(_N2, N2.data(), sizeof(float)* N2.size());
	cudaMemcpyToSymbol(_ion_SL_table, SL_binds_ion.data(), sizeof(short) * SL_binds_ion.size());
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
	cudaMemcpyToSymbol(_Jmax, Jmax.data(), sizeof(float)* Jmax.size());
	cudaMemcpyToSymbol(_Kserca, Kup.data(), sizeof(float)* Kup.size());
	cudaMemcpyToSymbol(_ion_SERCA_table, SR_uptakes_ion.data(), sizeof(short) * SR_uptakes_ion.size());
	//calculate multiplier for gradient
	pointers_are_set = false;
	dt_is_set = false;
	n_buffers = idx_buf;
	cudaMemcpyToSymbol(_n_buffers, &n_buffers, sizeof(n_buffers));
	n_buf_unique = buffers.size();
	cudaMemcpyToSymbol(_n_buffers_unique, &n_buf_unique, sizeof(n_buffers));
	n_blocks_init = (n_elements + threads_per_block - 1) / threads_per_block;
	std::vector<float> ions_b;
	for (auto const& i : ions)
		ions_b.push_back(i->Cb);
	cudaMemcpyToSymbol(_ionsb, ions_b.data(), sizeof(float) * ions_b.size());
	cudaMemcpyToSymbol(_D_ions, D_ions.data(), sizeof(float) * D_ions.size());
	cudaMemcpyToSymbol(_D_buf, D_buf.data(), sizeof(float) * D_buf.size());
	cudaMemcpyToSymbol(_kon, kon_buf.data(), sizeof(float) * kon_buf.size());
	cudaMemcpyToSymbol(_koff, koff_buf.data(), sizeof(float) * koff_buf.size());
	cudaMemcpyToSymbol(_buf_tot, total_buf.data(), sizeof(float) * total_buf.size());
	cudaMemcpyToSymbol(_ion_buffer_table, buffer_binds_ion.data(), sizeof(short) * buffer_binds_ion.size());
	cudaMemcpyToSymbol(_ion_SL_table, SL_binds_ion.data(), sizeof(short)* SL_binds_ion.size());
	cudaMalloc(&d_currents_grid, n_elements* n_ions * sizeof(float));
	cudaMalloc(&d_init_buffers, idx_buf * sizeof(float));
	cudaMemcpy(d_init_buffers, buf_init.data(), idx_buf * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_ions, n_elements * n_ions * sizeof(double));
	cudaMalloc(&d_ions_f, n_elements * n_ions * sizeof(float));
	cudaMalloc(&evo_ions, n_elements * n_ions * sizeof(float));
	cudaMalloc(&evo_buffers, n_elements * idx_buf * sizeof(float));
	cudaMalloc(&d_ions_near_channels, n_elements_near_channels * sizeof(double));
	cudaMalloc(&d_buffers, n_elements * idx_buf * sizeof(double));
	cudaMalloc(&d_buffers_f, n_elements * idx_buf * sizeof(float));
	cudaMalloc(&d_buffers_free, n_elements * n_buf_unique * sizeof(float));
	_SetupUpdateGraph();
}

void DyadRD2DwCytosol::_ZeroCurrents() {
	for (int i = 0; i < n_threads; ++i)
		for (int j = 0; j < n_channels; ++j)
			for (int k = 0; k < n_ions; ++k)
				ions_near_channels[j + n_channels * (i + k * n_threads)] = 0;
}

void DyadRD2DwCytosol::_SetupUpdateGraph() {
	cudaStreamBeginCapture(stream_dyad_cyto_2d, cudaStreamCaptureModeGlobal);
	get_ions_near_channels2DwCytosol << <n_threads, n_channels, 0, stream_dyad_cyto_2d >> > (d_ions, d_ions_near_channels);
	cudaMemcpyAsync(ions_near_channels.data(), d_ions_near_channels, n_elements_near_channels * sizeof(double), cudaMemcpyDeviceToHost, stream_dyad_cyto_2d);
	cudaStreamEndCapture(stream_dyad_cyto_2d, &update_graph_cyto_2d);
	cudaGraphInstantiate(&update_instance_cyto_2d, update_graph_cyto_2d, NULL, NULL, 0);
}

void DyadRD2DwCytosol::_SetupStepGraph() {
	cudaStreamBeginCapture(stream_dyad_cyto_2d, cudaStreamCaptureModeGlobal);
	evolution_operator << <grid, block, (n_ions + n_buffers) * (block.x +2) * (block.y + 2) * block.z * sizeof(float), stream_dyad_cyto_2d >> > (d_ions_f, d_buffers_f, evo_ions, evo_buffers, d_buffers_free, d_currents_grid);
	evo_step2DCyto << <grid, block, 0, stream_dyad_cyto_2d >> > (evo_ions, d_ions, d_ions_f, evo_buffers, d_buffers, d_buffers_f); //DOUBLE PRECISION KERNEL 
	get_free_buffers2DCyto << <grid, block, 0, stream_dyad_cyto_2d >> > (d_buffers_f, d_buffers_free);
	cudaStreamEndCapture(stream_dyad_cyto_2d, &timestep_graph_cyto_2d);
	cudaGraphInstantiate(&timestep_instance_cyto_2d, timestep_graph_cyto_2d, NULL, NULL, 0);
}

std::vector<double> DyadRD2DwCytosol::GaussElimination(std::vector<std::vector<double> >& a) {
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

double DyadRD2DwCytosol::GetElementVolume() {
	return dx * dy * z;
}

int DyadRD2DwCytosol::GetNumSRChannels() {
	return sr_size;
}
void DyadRD2DwCytosol::Reset() {
	set_initialsMP2DwCyto << <grid, block >> > (d_ions_f, d_buffers_f, d_init_buffers, d_currents_grid, d_buffers_free, d_ions, d_buffers);
	_ZeroCurrents();
}

bool DyadRD2DwCytosol::UsesGPU() {
	return true;
}

int DyadRD2DwCytosol::GetNIons() {
	return n_ions;
}

std::vector<double> DyadRD2DwCytosol::GetTotalSRCurrent() {
	return total_sr_current;
}

void DyadRD2DwCytosol::InitOpening(int thread, int channel) {
	channels->InitOpening(thread, channel);
}

void DyadRD2DwCytosol::_SetCurrents(const std::vector<double>& ions_jsr, const std::vector<double>& ions_extracell) {
	int idx, idx_jsr, idx_extra;
	for (int i = 0; i < n_threads; ++i) {
		total_sr_current[i] = 0;
		for (int j = 0; j < sr_size; ++j) {
			for (int k = 0; k < n_ions; ++k) {
				idx = j + n_channels * (i + k * n_threads);
				idx_jsr = j + sr_size * (i + k * n_threads);
				currents[idx] = channels->GetSRIonChannel(i, j)->Flux(ions_jsr[idx_jsr], ions_near_channels[idx], k);
				total_sr_current[i + k * n_threads] += currents[idx];
			}
		}
		for (int j = sr_size; j < n_channels; ++j) {
			for (int k = 0; k < n_ions; ++k) {
				idx = j + n_channels * (i + k * n_threads);
				idx_extra = j - sr_size + sl_size * (i + k * n_threads);
				currents[idx] = channels->GetSLIonChannel(i, j - sr_size)->Flux(ions_extracell[idx_extra], ions_near_channels[idx], k);
			}
		}
	}
	cudaMemcpyAsync(d_currents, currents.data(), currents.size() * sizeof(float), cudaMemcpyHostToDevice, stream_dyad_cyto_2d);
	set_currents2DCyto << <n_threads, n_channels, 0, stream_dyad_cyto_2d >> > (d_currents_grid, d_currents);
}

void DyadRD2DwCytosol::GetEffluxes(double*& currents_out) {
	currents_out = nullptr;
}

void DyadRD2DwCytosol::Update(double*& ions_cytosol, double*& buffers_cytosol, const std::vector<double>& jsr_ions, const std::vector<double>& extracellular_ions) {
	cudaGraphLaunch(update_instance_cyto_2d, stream_dyad_cyto_2d);
	_SetCurrents(jsr_ions, extracellular_ions);
}

//Iteration on calcium and buffers concentrations in dyad 
void DyadRD2DwCytosol::RunRD(double dt, int idx_b) {
	double dtt = dt;
	if (!dt_is_set) {
		cudaMemcpyToSymbol(_dt, &dtt, sizeof(double));
		_SetupStepGraph();
		dt_is_set = true;
	}
	cudaGraphLaunch(timestep_instance_cyto_2d, stream_dyad_cyto_2d);
}

void DyadRD2DwCytosol::RunMC(double dt, int n_thread) {
	channels->RunMC(dt, n_thread, ions_near_channels);
}

double DyadRD2DwCytosol::GetL() {
	return R;
}

std::map < std::string, std::vector<double> > DyadRD2DwCytosol::GetConcentrations(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions.size(); ++i) {
		auto it = std::find(values.begin(), values.end(), ions[i]->name);
		if (it != values.end()) {
			out.insert(std::pair<std::string, std::vector<double> >(ions[i]->name, std::vector<double>(n_elements)));
			cudaMemcpyAsync(out[ions[i]->name].data(), d_ions + i * n_elements, n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream_dyad_cyto_2d);
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
					cudaMemcpyAsync(out[type_name].data(), d_buffers + idx_b * n_elements, n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream_dyad_cyto_2d);
				}
			}
		}
	}
	cudaDeviceSynchronize();
	return out;
}

std::map < std::string, std::vector<int> > DyadRD2DwCytosol::GetChannelsStates(std::vector<std::string>& values) {
	std::map < std::string, std::vector<int> > out;
	std::string name = "RyR";
	std::vector<int> states;
	for (int i = 0; i < n_threads; ++i)
		for (int j = 0; j < n_channels; ++j)
			states.push_back(channels->GetSRIonChannel(i, j)->GetKineticModel()->GetState());
	out[name] = states;
	return out;
}

std::map <std::string, std::vector<double> > DyadRD2DwCytosol::GetIonsNearChannels(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions.size(); ++i) {
		auto itb = ions_near_channels.begin() + i * n_threads * n_channels;
		auto ite = ions_near_channels.begin() + (i + 1) * n_threads * n_channels;
		out.insert(std::pair<std::string, std::vector<double> >(ions[i]->name, std::vector<double>(itb, ite)));
	}
	return out;
}

std::vector<uint64_t> DyadRD2DwCytosol::GetDimensions() {
	return dyad_dims;
}

std::vector<uint64_t> DyadRD2DwCytosol::GetChannelsDimensions() {
	return channels_dims;
}
std::vector<uint64_t> DyadRD2DwCytosol::GetIonsNearChannelsDimensions() {
	return channels_ions_dims;
}

DyadRD2DwCytosol::~DyadRD2DwCytosol() {
	cudaFree(d_mult_z);
	cudaFree(&d_ions);
	cudaFree(&d_ions_f);
	cudaFree(&d_ions_near_channels);
	cudaFree(&d_ions_gradients);
	cudaFree(&evo_ions);
	cudaFree(&d_buffers);
	cudaFree(&d_buffers_f);
	cudaFree(&d_buffers_free);
	cudaFree(&d_buffers_near_channels);
	cudaFree(&d_init_buffers);
	cudaFree(&d_gradients_x);
	cudaFree(&d_gradients_y);
	cudaFree(&d_gradients);
	cudaFree(&d_buffers_gradients);
	cudaFree(&d_buffers_new);
	cudaFree(&d_buffer_binds_ion);
	cudaFree(&evo_buffers);
	cudaFree(&d_currents);
	cudaFree(&d_currents_grid);
	cudaFree(&d_init_buffers);
	cudaGraphExecDestroy(timestep_instance_cyto_2d);
	cudaGraphExecDestroy(update_instance_cyto_2d);
	cudaGraphDestroy(timestep_graph_cyto_2d);
	cudaGraphDestroy(update_graph_cyto_2d);
	cudaStreamDestroy(stream_dyad_cyto_2d);
}
