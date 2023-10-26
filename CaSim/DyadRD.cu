#include "DyadRD.h"

#include <algorithm> //find
#include <utility>

__device__ __constant__ double _dx;
__device__ __constant__ double _dy;
__device__ __constant__ double _dz;
__device__ __constant__ double _sx;
__device__ __constant__ double _sy;
__device__ __constant__ int _n_threads;
__device__ __constant__ int  _nx;
__device__ __constant__ int  _ny;
__device__ __constant__ int  _nz;
__device__ __constant__ int  _n_elements;
__device__ __constant__ int  _n_elements_per_thread;
__device__ __constant__ int  _n_ions_buffers;
__device__ __constant__ int _n_channels;
__device__ __constant__ int _n_ions;
__device__ __constant__ int _n_buffers; //number of buffers with diffusion
__device__ __constant__ int _n_buffers_unique; //total number of buffers
__device__ __constant__ int _n_threads_for_grad_x;
__device__ __constant__ int _n_threads_for_grad_y;
__device__ __constant__ double _grad_mult;
__device__ __constant__ double _dt;

cudaStream_t stream_dyad;
cudaGraph_t timestep_graph, update_graph;
cudaGraphExec_t timestep_instance, update_instance;

__device__ double d2YdX2(double Yl, double Yc, double Yr, double dX) {
	return (Yr - 2 * Yc + Yl) / (dX * dX);
}

__device__ int Index(int idx, int idy, int idz, int idw) {
	return idx + _nx * (idy + _ny * (idz + _nz * idw));
}

//initial values (Ca + buffers)
__global__ void set_initials(double* ions, double* init_ions, double* buffers, double* init_buffers, double* currents, double* buf_free, double* buf_tot, short* ion_buffer_table, double* gradients, double* ions_d, double* buffers_d) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idz = threadIdx.z + blockDim.z * blockIdx.z;
	int idx_batch = idz / _nz;
	idz %= _nz;
	int idfull = Index(idx, idy, idz, idx_batch);
	if (idx < _nx && idy < _ny && idz < _nz && idx_batch < _n_threads) {
		int _idx_buff = 0;
		for (int j = 0; j < _n_ions; ++j) {
			int _idx_ion = idfull + j * _n_elements;
			ions[_idx_ion] = init_ions[j];
			ions_d[_idx_ion] = (double)init_ions[j];
			currents[_idx_ion] = 0;
			for (int i = 0; i < _n_buffers_unique; ++i) {
				int idx_b = idfull + i * _n_elements;
				if (j == 0) {
					buf_free[idx_b] = buf_tot[i];
				}
				if (ion_buffer_table[i + j * _n_buffers_unique]) {
					double t = init_buffers[_idx_buff];
					buffers[idfull + _idx_buff * _n_elements] = t;
					buffers_d[idfull + _idx_buff * _n_elements] = (double)t;
					buf_free[idx_b] -= t;
					++_idx_buff;
				}
			}
		}
	}
	if (idfull < _n_ions_buffers * _n_threads)
		gradients[idfull] = 0;
}
// set currents values based on input double array and channels positions

__global__ void set_currents(double* current_grid, double* current_values, int* channel_x, int* channel_y, int* channel_z) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n_total_channels = _n_threads * _n_channels;
	int idx_grid = channel_x[threadIdx.x] + _nx * (channel_y[threadIdx.x] + _ny * (channel_z[threadIdx.x] + _nz * blockIdx.x));
	for (int j = 0; j < _n_ions; ++j)
		current_grid[idx_grid + j * _n_elements] = current_values[idx + j * n_total_channels];
}

__global__ void get_ions_near_channels(double* ions, double* ions_near_channels, int* channel_x, int* channel_y, int* channel_z) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n_total_channels = _n_threads * _n_channels;
	int idx_grid = channel_x[threadIdx.x] + _nx * (channel_y[threadIdx.x] + _ny * (channel_z[threadIdx.x] + _nz * blockIdx.x));
	for (int j = 0; j < _n_ions; ++j){
		ions_near_channels[idx + j * n_total_channels] = ions[idx_grid + j * _n_elements];
	}
}

__global__ void diffusion_step(double* ions, double* buffers, double* evo_ions, double* evo_buffers, double* D_ions, double* D_buf) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idz = threadIdx.z + blockDim.z * blockIdx.z;
	int idx_batch = idz / _nz;
	idz %= _nz;
	int idfull = Index(idx, idy, idz, idx_batch);
	if (idx < _nx && idy < _ny && idz < _nz && idx_batch < _n_threads) {
		if (!((idx == 0 || idx == _nx - 1) || (idy == 0 || idy == _ny - 1))) {
			int idxl = Index(idx - 1, idy, idz, idx_batch);
			int idxr = Index(idx + 1, idy, idz, idx_batch);
			int idyl = Index(idx, idy - 1, idz, idx_batch);
			int idyr = Index(idx, idy + 1, idz, idx_batch);
			int idzl = Index(idx, idy, idz - 1, idx_batch);
			int idzr = Index(idx, idy, idz + 1, idx_batch);
			double val_center;
			int str;
			if (idz == 0) {
				for (int j = 0; j < _n_ions; ++j) {
					str = j * _n_elements;
					val_center = ions[idfull + str];
					evo_ions[idfull + str] = D_ions[j] * (d2YdX2(ions[idxl + str], val_center, ions[idxr + str], _dx) +
						d2YdX2(ions[idyl + str], val_center, ions[idyr + str], _dy) +
						d2YdX2(ions[idzr + str], val_center, ions[idzr + str], _dz));
				}
				for (int i = 0; i < _n_buffers; ++i) {
					str = i * _n_elements;
					val_center = buffers[idfull + str];
					evo_buffers[idfull + str] = D_buf[i] * (d2YdX2(buffers[idxl + str], val_center, buffers[idxr + str], _dx) +
						d2YdX2(buffers[idyl + str], val_center, buffers[idyr + str], _dy) +
						d2YdX2(buffers[idzr + str], val_center, buffers[idzr + str], _dz));
				}
			}
			else if (idz == _nz - 1) {
				for (int j = 0; j < _n_ions; ++j) {
					str = j * _n_elements;
					val_center = ions[idfull + str];
					evo_ions[idfull + str] = D_ions[j] * (d2YdX2(ions[idxl + str], val_center, ions[idxr + str], _dx) +
						d2YdX2(ions[idyl + str], val_center, ions[idyr + str], _dy) +
						d2YdX2(ions[idzl + str], val_center, ions[idzl + str], _dz));
				}
				for (int i = 0; i < _n_buffers; ++i) {
					str = i * _n_elements;
					val_center = buffers[idfull + str];
					evo_buffers[idfull + str] = D_buf[i] * (d2YdX2(buffers[idxl + str], val_center, buffers[idxr + str], _dx) +
						d2YdX2(buffers[idyl + str], val_center, buffers[idyr + str], _dy) +
						d2YdX2(buffers[idzl + str], val_center, buffers[idzl + str], _dz));
				}
			}
			else {
				for (int j = 0; j < _n_ions; ++j) {
					str = j * _n_elements;
					val_center = ions[idfull + str];
					evo_ions[idfull + str] = D_ions[j] * (d2YdX2(ions[idxl + str], val_center, ions[idxr + str], _dx) +
						d2YdX2(ions[idyl + str], val_center, ions[idyr + str], _dy) +
						d2YdX2(ions[idzl + str], val_center, ions[idzr + str], _dz));
				}
				for (int i = 0; i < _n_buffers; ++i) {
					str = i * _n_elements;
					val_center = buffers[idfull + str];
					evo_buffers[idfull + str] = D_buf[i] * (d2YdX2(buffers[idxl + str], val_center, buffers[idxr + str], _dx) +
						d2YdX2(buffers[idyl + str], val_center, buffers[idyr + str], _dy) +
						d2YdX2(buffers[idzl + str], val_center, buffers[idzr + str], _dz));
				}
			}
		}
	}
}

__global__ void reaction_step(double* ions, double* buffers, double* evo_ions, double* evo_buffers, double* kon, double* koff, double* buf_free, short* ion_buffer_table) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idz = threadIdx.z + blockDim.z * blockIdx.z;
	int idx_batch = idz / _nz;
	idz %= _nz;
	int idfull = Index(idx, idy, idz, idx_batch);
	if (idx < _nx && idy < _ny && idz < _nz && idx_batch < _n_threads) {
		if (!((idx == 0 || idx == _nx - 1) || (idy == 0 || idy == _ny - 1))) {
			double R;
			int str, str2, str3;
			int _idx_buff = 0;
			double R1;
			for (int j = 0; j < _n_ions; ++j) {
				str = idfull + j * _n_elements;
				R = 0;
				for (int i = 0; i < _n_buffers_unique; ++i) {
					if (ion_buffer_table[i + j * _n_buffers_unique]) {
						str2 = idfull + _idx_buff * _n_elements;
						str3 = idfull + i * _n_elements;
						R1 = koff[_idx_buff] * buffers[str2] - kon[_idx_buff] * ions[str] * buf_free[str3];
						evo_buffers[str2] = -R1;
						R += R1;
						++_idx_buff;
					}
				}
				evo_ions[str] = R;
			}
		}
	}
}

__global__ void sum_step(double* evo_ions_diffusion, double* ions, double* evo_buffers_diffusion, double* evo_ions_reaction, double* evo_buffers_reaction, double* currents, double* evo_ions_total, double* evo_buffers_total, double* N1, double* K1, double* N2, double* K2, short* ion_SL_table) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idz = threadIdx.z + blockDim.z * blockIdx.z;
	int idx_batch = idz / _nz;
	idz %= _nz;
	int idfull = Index(idx, idy, idz, idx_batch);;
	int _idx_SL = 0;
	int idx2;
	if (idx < _nx && idy < _ny && idz < _nz && idx_batch < _n_threads) {
		if (!((idx == 0 || idx == _nx - 1) || (idy == 0 || idy == _ny - 1))) {
			for (int j = 0; j < _n_ions; ++j) {
				idx2 = idfull + j * _n_elements;
				evo_ions_total[idx2] = evo_ions_diffusion[idx2] + evo_ions_reaction[idx2] + currents[idx2];
				if (idz == _nz - 1 && ion_SL_table[j]) {
					double s1 = K1[_idx_SL] + ions[idx2];
					double s2 = K2[_idx_SL] + ions[idx2];
					evo_ions_total[idx2] *= 1 / (1 + N1[_idx_SL] * K1[_idx_SL] / (s1 * s1) + N2[_idx_SL] * K2[_idx_SL] / (s2 * s2));
					++_idx_SL;
				}
			}
			for (int i = 0; i < _n_buffers; ++i) {
				idx2 = idfull + i * _n_elements;
				evo_buffers_total[idx2] = evo_buffers_diffusion[idx2] + evo_buffers_reaction[idx2];
			}
		}
	}
}

__global__ void evo_step(double* evo_ions_total, double* ions, double* ions_f, double* evo_buffers_total, double* buffers, double* buffers_f, short* ion_buffer_table) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idz = threadIdx.z + blockDim.z * blockIdx.z;
	int idx_batch = idz / _nz;
	idz %= _nz;
	int idfull = Index(idx, idy, idz, idx_batch);
	double tot_double;
	if (idx < _nx && idy < _ny && idz < _nz && idx_batch < _n_threads) {
		if (!((idx == 0 || idx == _nx - 1) || (idy == 0 || idy == _ny - 1))) {
			int idx2;
			for (int j = 0; j < _n_ions; ++j) {
				idx2 = idfull + j * _n_elements;
				tot_double = (double)evo_ions_total[idx2];
				ions[idx2] +=  tot_double * _dt;
				ions_f[idx2] = ions[idx2];
			}
			for (int i = 0; i < _n_buffers; ++i) {
				idx2 = idfull + i * _n_elements;
				tot_double = (double)evo_buffers_total[idx2];
				buffers[idx2] += + tot_double * _dt;
				buffers_f[idx2] = buffers[idx2];
			}
		}
	}
}


__global__ void get_free_buffers(double* buffers, double* buf_free, double* buf_tot, short* ion_buffer_table) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idz = threadIdx.z + blockDim.z * blockIdx.z;
	int idx_batch = idz / _nz;
	idz %= _nz;
	int idfull = Index(idx, idy, idz, idx_batch);
	if (idx < _nx && idy < _ny && idz < _nz && idx_batch < _n_threads) {
		if (!((idx == 0 || idx == _nx - 1) || (idy == 0 || idy == _ny - 1))) {
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
}


//Gradients dimensions: x - x coordinate, block - scenario
__global__ void gradient_x(double* ions, double* buffers, short* ion_buffer_table, double* grads_x, double* D_ions, double* D_buffers, double* mult) {
	int idx = threadIdx.x;
	int n_batch = blockIdx.x;
	int blocksize = blockDim.x;
	extern __shared__ double gradients_x[];
	for (int i = 0; i < _n_ions_buffers; ++i) {
		gradients_x[idx + blocksize * i] = 0;
	}
	__syncthreads();
	int str = 0;
	if (idx > 0 && idx < _nx) {
		int idfull = 0;
		int idfull2 = 0;
		int idfull3 = 0;
		int idy; //number of blocks == number of threads == number of simulations in batch
		str = 0;
		for (int idz = 0; idz < _nz; ++idz) {
			idy = 1;
			idfull = Index(idx, idy, idz, n_batch);
			idy = 0;
			idfull2 = Index(idx, idy, idz, n_batch);
			idy = 2;
			idfull3 = Index(idx, idy, idz, n_batch);
			for (int j = 0; j < _n_ions; ++j) {
				str = j * _n_elements;
				gradients_x[idx + blocksize * j] += mult[idz] * (2 * ions[idfull + str] - 1.5 * ions[idfull2 + str] - 0.5 * ions[idfull3 + str]);
			}
			for (int i = 0; i < _n_buffers; ++i) {
				str = i * _n_elements;
				gradients_x[idx + blocksize * (_n_ions + i)] += mult[idz] * (2 * buffers[idfull + str] - 1.5 * buffers[idfull2 + str] - 0.5 * buffers[idfull3 + str]);
			}
		}
		for (int idz = 0; idz < _nz; ++idz) {
			idy = _ny - 2;
			idfull = Index(idx, idy, idz, n_batch);
			idy = _ny - 1;
			idfull2 = Index(idx, idy, idz, n_batch);
			idy = _ny - 3;
			idfull3 = Index(idx, idy, idz, n_batch);
			for (int j = 0; j < _n_ions; ++j) {
				str = j * _n_elements;
				gradients_x[idx + blocksize * j] += mult[idz] * (2 * ions[idfull + str] - 1.5 * ions[idfull2 + str] - 0.5 * ions[idfull3 + str]);
			}
			for (int i = 0; i < _n_buffers; ++i) {
				str = i * _n_elements;
				gradients_x[idx + blocksize * (_n_ions + i)] += mult[idz] * (2 * buffers[idfull + str] - 1.5 * buffers[idfull2 + str] - 0.5 * buffers[idfull3 + str]);
			}
		}
	}
	__syncthreads();
	//make a sum of all values in block
	for (int div = blocksize / 2; div > 0; div /= 2) {
		if (idx < div) {
			for (int j = 0; j < _n_ions; ++j) {
				str = j * _n_elements;
				gradients_x[idx + blocksize * j] += gradients_x[idx + div + blocksize * j];
			}
			for (int i = 0; i < _n_buffers; ++i) {
				gradients_x[idx + blocksize * (_n_ions + i)] += gradients_x[idx + div + blocksize * (_n_ions + i)];
			}
		}
		__syncthreads();
	}
	//set values to specific array
	if (idx == 0) {
		for (int j = 0; j < _n_ions; ++j) 
			grads_x[j + _n_ions_buffers * n_batch] = D_ions[j] * _grad_mult * gradients_x[blocksize * j] * _sx;
		for (int i = 0; i < _n_buffers; ++i)
			grads_x[i + _n_ions + _n_ions_buffers * n_batch] = D_buffers[i] * _grad_mult * gradients_x[blocksize * (_n_ions + i)] * _sx;
	}
}

__global__ void gradient_y(double* ions, double* buffers, short* ion_buffer_table, double* grads_x, double* D_ions, double* D_buffers,double* mult) {
	int idy = threadIdx.x;
	int n_batch = blockIdx.x;
	int blocksize = blockDim.x;
	extern __shared__ double gradients_x[];
	for (int i = 0; i < _n_ions_buffers; ++i) {
		gradients_x[threadIdx.x + blocksize * i] = 0;
	}
	__syncthreads();
	int str = 0;
	if (idy > 0 && idy < _ny) {
		int idfull = 0;
		int idfull2 = 0;
		int idfull3 = 0;
		int idx; //number of blocks == number of threads == number of simulations in batch
		str = 0;
		for (int idz = 0; idz < _nz; ++idz) {
			idx = 1;
			idfull = Index(idx, idy, idz, n_batch);
			idx = 0;
			idfull2 = Index(idx, idy, idz, n_batch);
			idx = 2;
			idfull3 = Index(idx, idy, idz, n_batch);
			for (int j = 0; j < _n_ions; ++j) {
				str = j * _n_elements;
				gradients_x[threadIdx.x + blocksize * j] += mult[idz] * (2 * ions[idfull + str] - 1.5 * ions[idfull2 + str] - 0.5 * ions[idfull3 + str]);
			}
			for (int i = 0; i < _n_buffers; ++i) {
				str = i * _n_elements;
				gradients_x[threadIdx.x + blocksize * (_n_ions + i)] += mult[idz] * (2 * buffers[idfull + str] - 1.5 * buffers[idfull2 + str] - 0.5 * buffers[idfull3 + str]);
			}
		}
		for (int idz = 0; idz < _nz; ++idz) {
			idx = _nx - 2;
			idfull = Index(idx, idy, idz, n_batch);
			idx = _nx - 1;
			idfull2 = Index(idx, idy, idz, n_batch);
			idx = _nx - 3;
			idfull3 = Index(idx, idy, idz, n_batch);
			for (int j = 0; j < _n_ions; ++j) {
				str = j * _n_elements;
				gradients_x[threadIdx.x + blocksize * j] += mult[idz] * (2 * ions[idfull + str] - 1.5 * ions[idfull2 + str] - 0.5 * ions[idfull3 + str]);
			}
			for (int i = 0; i < _n_buffers; ++i) {
				str = i * _n_elements;
				gradients_x[threadIdx.x + blocksize * (_n_ions + i)] += mult[idz] * (2 * buffers[idfull + str] - 1.5 * buffers[idfull2 + str] - 0.5 * buffers[idfull3 + str]);
			}
		}
	}
	__syncthreads();
	//make a sum of all values in block
	for (int div = blocksize / 2; div > 0; div /= 2) {
		if (idy < div) {
			for (int j = 0; j < _n_ions; ++j) {
				str = j * _n_elements;
				gradients_x[idy + blocksize * j] += gradients_x[idy + div + blocksize * j];
			}
			for (int i = 0; i < _n_buffers; ++i) {
				gradients_x[idy + blocksize * (_n_ions + i)] += gradients_x[idy + div + blocksize * (_n_ions + i)];
			}
		}
		__syncthreads();
	}
	//set values to specific array
	if (idy == 0) {
		for (int j = 0; j < _n_ions; ++j)
			grads_x[j + _n_ions_buffers * n_batch] = D_ions[j] * _grad_mult * gradients_x[blocksize * j] * _sy;
		for (int i = 0; i < _n_buffers; ++i)
			grads_x[i + _n_ions + _n_ions_buffers * n_batch] = D_buffers[i] * _grad_mult * gradients_x[blocksize * (_n_ions + i)] * _sy;
	}
}

__global__ void gradient_sum(double* grads_x, double* grads_y, double* grads) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	grads[idx] = grads_x[idx] + grads_y[idx];
}

__global__ void set_boundaries(double* ions, double* buffers, double* ions_boundary, double* buffers_boundary) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idz = threadIdx.z + blockDim.z * blockIdx.z;
	int idx_batch = idz / _nz;
	idz %= _nz;
	int idfull = Index(idx, idy, idz, idx_batch);
	if (idx < _nx && idy < _ny && idz < _nz && idx_batch < _n_threads) {
		if (((idx == 0 || idx == _nx - 1) || (idy == 0 || idy == _ny - 1))) {
			int idx2;
			for (int j = 0; j < _n_ions; ++j) {
				idx2 = idfull + j * _n_elements;
				ions[idx2] = ions_boundary[j + idx_batch * _n_ions];
			}
			for (int i = 0; i < _n_buffers; ++i) {
				idx2 = idfull + i * _n_elements;
				buffers[idx2] = buffers_boundary[i + idx_batch * _n_buffers];
			}
		}
	}
}


DyadRD::DyadRD(nlohmann::json& j, int nthreads) {
	channels = std::make_unique<DyadChannels>(j, nthreads);
	dx = j["dx"];
	cudaMemcpyToSymbol(_dx, &dx, sizeof(double));
	dy = j["dy"];
	cudaMemcpyToSymbol(_dy, &dy, sizeof(double));
	dz = j["dz"];
	cudaMemcpyToSymbol(_dz, &dz, sizeof(double));
	x = j["x"];
	y = j["y"];
	z = j["z"];
	blockx = j["CUDA"]["BLOCK X"];
	blocky = j["CUDA"]["BLOCK Y"];
	blockz = j["CUDA"]["BLOCK Z"];
	R = j["Radius"];
	block.x = blockx, block.y = blocky, block.z = blockz;
	n_threads = nthreads;
	cudaStreamCreate(&(stream_dyad));
	cudaMemcpyToSymbol(_n_threads, &n_threads, sizeof(n_threads));
	nx = x / dx + 1;
	cudaMemcpyToSymbol(_nx, &nx, sizeof(nx));
	ny = y / dy + 1;
	cudaMemcpyToSymbol(_ny, &ny, sizeof(ny));
	nz = z / dz + 1;
	cudaMemcpyToSymbol(_nz, &nz, sizeof(nz));
	dyad_dims.resize(5);
	dyad_dims[1] = nx;
	dyad_dims[2] = ny;
	dyad_dims[3] = nz;
	dyad_dims[4] = n_threads;
	double sx = dy * dz / dx;
	double sy = dx * dz / dy;
	std::vector<double> mult_z;
	for (int i = 0; i < nz; ++i) {
		if (i == 0 || i == nz - 1)
			mult_z.push_back(0.5);
		else
			mult_z.push_back(1.0);
	}
	cudaMalloc(&d_mult_z, nz * sizeof(double));
	cudaMemcpy(d_mult_z, mult_z.data(), mult_z.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_sx, &sx, sizeof(sx));
	cudaMemcpyToSymbol(_sy, &sy, sizeof(sy));
	int n_elements_thread = nx * ny * nz;
	cudaMemcpyToSymbol(_n_elements_per_thread, &n_elements_thread, sizeof(n_elements_thread));
	grid.x = (nx + block.x - 1) / blockx, grid.y = (ny + block.y - 1) / blocky, grid.z = (nz * n_threads + block.z - 1) / blockz;
	n_elements = nx * ny * nz * n_threads;
	cudaMemcpyToSymbol(_n_elements, &n_elements, sizeof(n_elements));
	n_threads_grad_x = 1;
	while (n_threads_grad_x < nx)
		n_threads_grad_x *= 2;
	cudaMemcpyToSymbol(_n_threads_for_grad_x, &n_threads_grad_x, sizeof(n_threads_grad_x));
	n_threads_grad_y = 1;
	while (n_threads_grad_y < ny)
		n_threads_grad_y *= 2;
	cudaMemcpyToSymbol(_n_threads_for_grad_y, &n_threads_grad_y, sizeof(n_threads_grad_y));
	//Kernels set 256 threads per block, number of blocks depends on the number of elements
	threads_per_block = block.x * block.y * block.z;
	//Ions
	n_ions = 0;
	for (auto const& el : j["Ions"].items()) {
		auto pars_ion = el.value();
		ions.push_back(std::make_unique<Ion<double> >(el.key(), pars_ion["D"], pars_ion["Concentration"]));
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
	for (int j = 0; j < sr_size; ++j) {
		auto p = channels->GetSRIonChannel(0, j)->GetCoordinates()->GetCoordsOnGrid(dx, dy, dz);
		x_idx_v.push_back(p[0]);
		y_idx_v.push_back(p[1]);
		z_idx_v.push_back(p[2]);
	}
	for (int j = sr_size; j < n_channels; ++j) {
		auto p = channels->GetSLIonChannel(0, j - sr_size)->GetCoordinates()->GetCoordsOnGrid(dx, dy, dz);
		x_idx_v.push_back(p[0]);
		y_idx_v.push_back(p[1]);
		z_idx_v.push_back(p[2]);
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
	cudaMalloc(&d_currents, n_channels * n_ions * n_threads * sizeof(double));
	cudaMalloc(&x_idx, x_idx_v.size() * sizeof(int));
	cudaMemcpy(x_idx, x_idx_v.data(), x_idx_v.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&y_idx, y_idx_v.size() * sizeof(int));
	cudaMemcpy(y_idx, y_idx_v.data(), y_idx_v.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&z_idx, z_idx_v.size() * sizeof(int));
	cudaMemcpy(z_idx, z_idx_v.data(), z_idx_v.size() * sizeof(int), cudaMemcpyHostToDevice);
	ions_near_channels.resize(n_elements_near_channels);
	cudaMemcpyToSymbol(_n_ions, &n_ions, sizeof(n_ions));
	//Buffers
	for (auto const& el : j["Buffers"].items()) {
		auto pars_buff = el.value();
		buffers.push_back(std::make_unique<Buffer<double> >(el.key(), pars_buff["Total Concentration"]));
		total_buf.push_back(pars_buff["Total Concentration"]);
		for (auto const& el_ion : pars_buff["Ions"].items()) {
			auto pars_buff_ion = el_ion.value();
			buffers[buffers.size() - 1]->ions_kinetics[el_ion.key()] = std::make_unique<BufferIon<double> >(pars_buff_ion["D"], pars_buff_ion["kon"], pars_buff_ion["koff"]);
		}
		int n_ions_buf = buffers[buffers.size() - 1]->ions_kinetics.size();
		std::vector<double> zeroes(n_ions_buf + 1, 0);
		std::vector<std::vector<double> > a(n_ions_buf,zeroes);
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
			}
			else
				buffer_binds_ion.push_back(0);
		}
	for (auto const& el : j["Sarcolemma"].items()) {
		auto pars_SL = el.value();
		ions_and_SL[el.key()] = std::make_unique<IonSarcolemma<double> >(pars_SL["N1"], pars_SL["K1"], pars_SL["N2"], pars_SL["K2"]);
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
	//calculate multiplier for gradient
	pointers_are_set = false;
	dt_is_set = false;
	grad_mult =  3.0f / (4 * acos(-1) * R * R * R);
	cudaMemcpyToSymbol(_grad_mult, &grad_mult, sizeof(grad_mult));
	n_buffers = idx_buf;
	cudaMemcpyToSymbol(_n_buffers, &n_buffers, sizeof(n_buffers));
	int n_buf_unique = buffers.size();
	cudaMemcpyToSymbol(_n_buffers_unique, &n_buf_unique, sizeof(n_buffers));
	n_blocks_init = (n_elements + threads_per_block - 1) / threads_per_block;
	int n_ions_and_buffers = idx_buf + n_ions;
	cudaMemcpyToSymbol(_n_ions_buffers, &n_ions_and_buffers, sizeof(n_ions_and_buffers));
	cudaMalloc(&d_gradients_x, (idx_buf + n_ions) *  n_threads * sizeof(double));
	cudaMalloc(&d_gradients_y, (idx_buf + n_ions) * n_threads * sizeof(double));
	cudaMalloc(&d_gradients, (idx_buf + n_ions) * n_threads * sizeof(double));
	cudaMalloc(&d_ions, n_elements * n_ions * sizeof(double));
	cudaMalloc(&d_ions_f, n_elements * n_ions * sizeof(double));
	cudaMalloc(&evo_ions_diffusion, n_elements* n_ions * sizeof(double));
	cudaMalloc(&evo_buffers_diffusion, n_elements* idx_buf * sizeof(double));
	cudaMalloc(&evo_ions_reaction, n_elements* n_ions * sizeof(double));
	cudaMalloc(&evo_buffers_reaction, n_elements * idx_buf * sizeof(double));
	cudaMalloc(&evo_ions_total, n_elements * n_ions * sizeof(double));
	cudaMalloc(&evo_buffers_total, n_elements * idx_buf * sizeof(double));
	cudaMalloc(&d_ions_near_channels, n_elements_near_channels * sizeof(double));
	std::vector<double> ions_b;
	for (auto const& i : ions)
		ions_b.push_back(i->Cb);
	cudaMalloc(&d_init_ions, n_ions * sizeof(double));
	cudaMemcpy(d_init_ions, ions_b.data(), n_ions * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_ions_gradients, n_ions * n_threads * sizeof(double));
	cudaMalloc(&d_buffers_gradients, idx_buf * n_threads * sizeof(double));
	cudaMalloc(&d_buffers, n_elements * idx_buf * sizeof(double));
	cudaMalloc(&d_buffers_f, n_elements * idx_buf * sizeof(double));
	cudaMalloc(&d_buffers_free, n_elements * n_buf_unique * sizeof(double));
	cudaMalloc(&d_buffers_near_channels, n_channels * n_threads * idx_buf * sizeof(double));
	cudaMalloc(&d_D_ions, D_ions.size() * sizeof(double));
	cudaMemcpy(d_D_ions, D_ions.data(), D_ions.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_D_buf, D_buf.size() * sizeof(double));
	cudaMemcpy(d_D_buf, D_buf.data(), D_buf.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_kon, kon_buf.size() * sizeof(double));
	cudaMemcpy(d_kon, kon_buf.data(), kon_buf.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_koff, koff_buf.size() * sizeof(double));
	cudaMemcpy(d_koff, koff_buf.data(), koff_buf.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_CTot, total_buf.size() * sizeof(double));
	cudaMemcpy(d_CTot, total_buf.data(), total_buf.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_buffer_binds_ion, buffer_binds_ion.size() * sizeof(short));
	cudaMemcpy(d_buffer_binds_ion, buffer_binds_ion.data(), buffer_binds_ion.size() * sizeof(short), cudaMemcpyHostToDevice);
	cudaMalloc(&d_SL_binds_ion, SL_binds_ion.size() * sizeof(short));
	cudaMemcpy(d_SL_binds_ion, SL_binds_ion.data(), SL_binds_ion.size() * sizeof(short), cudaMemcpyHostToDevice);
	cudaMalloc(&d_currents_grid, n_elements * n_ions * sizeof(double));
	cudaMalloc(&d_init_buffers, idx_buf * sizeof(double));
	cudaMemcpy(d_init_buffers, buf_init.data(), idx_buf * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_K1, K1.size() * sizeof(double));
	cudaMemcpy(d_K1, K1.data(), K1.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_K2, K2.size() * sizeof(double));
	cudaMemcpy(d_K2, K2.data(), K2.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_N1, N1.size() * sizeof(double));
	cudaMemcpy(d_N1, N1.data(), N1.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&d_N2, N2.size() * sizeof(double));
	cudaMemcpy(d_N2, N2.data(), N2.size() * sizeof(double), cudaMemcpyHostToDevice);
}

void DyadRD::_ZeroCurrents() {
	for (int i = 0; i < n_threads; ++i)
		for (int j = 0; j < n_channels; ++j)
			for (int k = 0; k < n_ions; ++k)
				ions_near_channels[j + n_channels * (i + k * n_threads)] = 0;
}

void DyadRD::_SetupUpdateGraph() {
	cudaStreamBeginCapture(stream_dyad, cudaStreamCaptureModeGlobal);
	gradient_x << <n_threads, n_threads_grad_x, n_threads_grad_x* (idx_buf + n_ions) * sizeof(double), stream_dyad >> > (d_ions_f, d_buffers_f, d_buffer_binds_ion, d_gradients_x, d_D_ions, d_D_buf, d_mult_z);
	gradient_y << <n_threads, n_threads_grad_y, n_threads_grad_y* (idx_buf + n_ions) * sizeof(double), stream_dyad >> > (d_ions_f, d_buffers_f, d_buffer_binds_ion, d_gradients_y, d_D_ions, d_D_buf, d_mult_z);
	gradient_sum << <1, n_threads* (idx_buf + n_ions),0, stream_dyad >> > (d_gradients_x, d_gradients_y, d_gradients);
	get_ions_near_channels << <n_threads, n_channels, 0, stream_dyad >> > (d_ions_f, d_ions_near_channels, x_idx, y_idx, z_idx);
	cudaMemcpyAsync(ions_near_channels.data(), d_ions_near_channels, n_elements_near_channels * sizeof(double), cudaMemcpyDeviceToHost, stream_dyad);
	cudaStreamEndCapture(stream_dyad, &update_graph);
	cudaGraphInstantiate(&update_instance, update_graph, NULL, NULL, 0);
}

void DyadRD::_SetupStepGraph() {
	cudaStreamBeginCapture(stream_dyad, cudaStreamCaptureModeGlobal);
	diffusion_step << <grid, block, 0, stream_dyad >> > (d_ions_f, d_buffers_f, evo_ions_diffusion, evo_buffers_diffusion, d_D_ions, d_D_buf);
	reaction_step << <grid, block, 0, stream_dyad >> > (d_ions_f, d_buffers_f, evo_ions_reaction, evo_buffers_reaction, d_kon, d_koff, d_buffers_free, d_buffer_binds_ion);
	sum_step << <grid, block, 0, stream_dyad >> > (evo_ions_diffusion, d_ions_f, evo_buffers_diffusion, evo_ions_reaction, evo_buffers_reaction, d_currents_grid, evo_ions_total, evo_buffers_total, d_N1, d_K1, d_N2, d_K2, d_SL_binds_ion);
	evo_step << <grid, block, 0, stream_dyad >> > (evo_ions_total, d_ions, d_ions_f, evo_buffers_total, d_buffers, d_buffers_f, d_buffer_binds_ion); //DOUBLE PRECISION KERNEL 
	get_free_buffers << <grid, block, 0, stream_dyad >> > (d_buffers_f, d_buffers_free, d_CTot, d_buffer_binds_ion);
	cudaStreamEndCapture(stream_dyad, &timestep_graph);
	cudaGraphInstantiate(&timestep_instance, timestep_graph, NULL, NULL, 0);
}

std::vector<double> DyadRD::GaussElimination(std::vector<std::vector<double> >& a) {
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

double DyadRD::GetElementVolume() {
	return dx * dy * dz;
}

int DyadRD::GetNumSRChannels() {
	return sr_size;
}
void DyadRD::Reset() {
	set_initials << <grid, block >> > (d_ions_f, d_init_ions, d_buffers_f, d_init_buffers, d_currents_grid, d_buffers_free, d_CTot, d_buffer_binds_ion, d_gradients, d_ions, d_buffers);
	_ZeroCurrents();
}

bool DyadRD::UsesGPU() {
	return true;
}

int DyadRD::GetNIons() {
	return n_ions;
}

std::vector<double> DyadRD::GetTotalSRCurrent() {
	return total_sr_current;
}

void DyadRD::InitOpening(int thread, int channel) {
	channels->InitOpening(thread, channel);
}

void DyadRD::_SetCurrents(const std::vector<double>& ions_jsr, const std::vector<double>& ions_extracell) {
	int idx, idx_jsr, idx_extra;
	for (int i = 0; i < n_threads; ++i) {
		total_sr_current[i] = 0;
		for (int j = 0; j < sr_size; ++j) {
			for (int k = 0; k < n_ions; ++k) {
				idx = j + n_channels * (i + k * n_threads);
				idx_jsr = j + sr_size * (i + k * n_threads);
				currents[idx] = channels->GetSRIonChannel(i, j)->Flux(ions_jsr[idx_jsr], ions_near_channels[idx],k);
				total_sr_current[i + k * n_threads] += currents[idx];
			}
		}
		for (int j = sr_size; j < n_channels; ++j) {
			for (int k = 0; k < n_ions; ++k) {
				idx = j + n_channels * (i + k * n_threads);
				idx_extra = j - sr_size + sl_size * (i + k * n_threads);
				currents[idx] = channels->GetSLIonChannel(i, j - sr_size)->Flux(ions_extracell[idx_extra], ions_near_channels[idx],k);
			}
		}
	}
	cudaMemcpyAsync(d_currents, currents.data(), currents.size() * sizeof(double), cudaMemcpyHostToDevice, stream_dyad);
	set_currents << <n_threads, n_channels, 0, stream_dyad >> > (d_currents_grid, d_currents, x_idx, y_idx, z_idx);
}

void DyadRD::GetEffluxes(double*& currents_out) {
	currents_out = d_gradients;
}

void DyadRD::Update(double*& ions_cytosol, double*& buffers_cytosol, const std::vector<double>& jsr_ions, const std::vector<double>& extracellular_ions) {//add reinit
	if (ions_from_cytosol != ions_cytosol || buffers_from_cytosol != buffers_cytosol) {
		ions_from_cytosol = ions_cytosol;
		buffers_from_cytosol = buffers_cytosol;
		_SetupStepGraph();
		_SetupUpdateGraph();
	}
	set_boundaries << <grid, block, 0, stream_dyad >> > (d_ions_f, d_buffers_f, ions_from_cytosol, buffers_from_cytosol);
	cudaGraphLaunch(update_instance, stream_dyad);
	_SetCurrents(jsr_ions, extracellular_ions);
}

//Iteration on calcium and buffers concentrations in dyad 
void DyadRD::RunRD(double dt, int idx_b) {
	double dtt = dt;
	if (!dt_is_set) {
		cudaMemcpyToSymbol(_dt, &dtt, sizeof(double));
		dt_is_set = true;
	}
	cudaGraphLaunch(timestep_instance, stream_dyad);
}

void DyadRD::RunMC(double dt, int n_thread) {
	channels->RunMC(dt, n_thread, ions_near_channels);
}

double DyadRD::GetL() {
	return R;
}

std::map < std::string, std::vector<double> > DyadRD::GetConcentrations(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions.size(); ++i) {
		auto it = std::find(values.begin(), values.end(), ions[i]->name);
		if (it != values.end()) {
			out.insert(std::pair<std::string, std::vector<double> >(ions[i]->name,std::vector<double>(n_elements)));
			cudaMemcpyAsync(out[ions[i]->name].data(), d_ions_f + i * n_elements, n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream_dyad);
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
					cudaMemcpyAsync(out[type_name].data(), d_buffers_f + idx_b * n_elements, n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream_dyad);
				}
			}
		}
	}
	cudaDeviceSynchronize();
	return out;
}

std::map < std::string, std::vector<int> > DyadRD::GetChannelsStates(std::vector<std::string>& values) {
	std::map < std::string, std::vector<int> > out;
	std::string name = "RyR";
	std::vector<int> states;
	for (int i = 0; i < n_threads; ++i)
		for (int j = 0; j < n_channels; ++j)
			states.push_back(channels->GetSRIonChannel(i, j)->GetKineticModel()->GetState());
	out[name] = states;
	return out;
}

std::map <std::string, std::vector<double> > DyadRD::GetIonsNearChannels(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions.size(); ++i) {
		auto itb = ions_near_channels.begin() + i * n_threads * n_channels;
		auto ite = ions_near_channels.begin() + (i + 1) * n_threads * n_channels;
		out.insert(std::pair<std::string, std::vector<double> >(ions[i]->name, std::vector<double>(itb,ite)));
	}
	return out;
}

std::vector<uint64_t> DyadRD::GetDimensions() {
	return dyad_dims;
}

std::vector<uint64_t> DyadRD::GetChannelsDimensions() {
	return channels_dims;
}
std::vector<uint64_t> DyadRD::GetIonsNearChannelsDimensions() {
	return channels_ions_dims;
}

DyadRD::~DyadRD() {
	cudaFree(&d_init_ions);
	cudaFree(&d_ions);
	cudaFree(&d_ions_f);
	cudaFree(&d_ions_near_channels);
	cudaFree(&d_ions_gradients);
	cudaFree(&d_D_ions);
	cudaFree(&evo_ions_diffusion);
	cudaFree(&evo_ions_reaction);
	cudaFree(&evo_ions);
	cudaFree(&evo_ions_total);
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
	cudaFree(&evo_buffers_diffusion);
	cudaFree(&evo_buffers_reaction);
	cudaFree(&evo_buffers);
	cudaFree(&evo_buffers_total);
	cudaFree(&d_kon);
	cudaFree(&d_koff);
	cudaFree(&d_D_buf);
	cudaFree(&d_CTot);
	cudaFree(&d_K1);
	cudaFree(&d_K2);
	cudaFree(&d_N1);
	cudaFree(&d_N2);
	cudaFree(&d_SL_binds_ion);
	cudaFree(&d_currents);
	cudaFree(&d_currents_grid);
	cudaFree(&d_init_buffers);
	cudaFree(&x_idx);
	cudaFree(&y_idx);
	cudaFree(&z_idx);
	cudaFree(&d_mult_z);
	cudaGraphExecDestroy(timestep_instance);
	cudaGraphExecDestroy(update_instance);
	cudaGraphDestroy(timestep_graph);
	cudaGraphDestroy(update_graph);
	cudaStreamDestroy(stream_dyad);
}