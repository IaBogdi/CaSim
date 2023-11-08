#include "DyadRDMP2D.h"

#include <algorithm> //find
#include <utility>

const int MAX_IONS = 10;
const int MAX_BUFFERS = 200;
const int MAX_CHANNELS = 100;
__device__ __constant__ float _dx;
__device__ __constant__ float _dy;
__device__ __constant__ double _sx;
__device__ __constant__ double _sy;
__device__ __constant__ int _n_threads;
__device__ __constant__ int  _nx;
__device__ __constant__ int  _ny;
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
__device__ __constant__ short _ion_buffer_table[MAX_BUFFERS];
__device__ __constant__ int _channel_x[MAX_CHANNELS];
__device__ __constant__ int _channel_y[MAX_CHANNELS];
__device__ __constant__ float _D_ions[MAX_IONS];
__device__ __constant__ float _D_buf[MAX_BUFFERS];
__device__ __constant__ float _kon[MAX_BUFFERS];
__device__ __constant__ float _koff[MAX_BUFFERS];
__device__ __constant__ float _buf_tot[MAX_BUFFERS];
__device__ __constant__ short _ion_SL_table[MAX_IONS];
__device__ __constant__ float _ionsb[MAX_IONS];
__device__ __constant__ float _K1[MAX_IONS];
__device__ __constant__ float _K2[MAX_IONS];
__device__ __constant__ float _N1[MAX_IONS];
__device__ __constant__ float _N2[MAX_IONS];

cudaStream_t stream_dyad_mp2d, stream_dyad_2_mp2d;
cudaEvent_t grad_event, grad_event2;
cudaGraph_t timestep_graph_mp2d, update_graph_mp2d;
cudaGraphExec_t timestep_instance_mp2d, update_instance_mp2d;

__device__ __forceinline__ float d2YdX2MP2D(float Yl, float Yc, float Yr, float dX) {
	return (Yr - 2 * Yc + Yl) / (dX * dX);
}

__device__ __forceinline__ int IndexMP2D(int idx, int idy, int idw) {
	return idy + _ny * (idx + _nx * idw);
}

//initial values (Ca + buffers)
__global__ void set_initialsMP2D(float* ions, float* buffers, float* init_buffers, float* currents, float* buf_free,double* gradients, double* ions_d, double* buffers_d) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_batch = threadIdx.z + blockDim.z * blockIdx.z;
	int idfull = IndexMP2D(idx, idy, idx_batch);
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
	if (idfull < _n_ions_buffers * _n_threads)
		gradients[idfull] = 0;
}
// set currents values based on input float array and channels positions

__global__ void set_currentsMP2D(float* current_grid, float* current_values) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n_total_channels = _n_threads * _n_channels;
	int idx_grid = IndexMP2D(_channel_x[threadIdx.x], _channel_y[threadIdx.x], blockIdx.x);
	for (int j = 0; j < _n_ions; ++j)
		current_grid[idx_grid + j * _n_elements] = current_values[idx + j * n_total_channels];
}

__global__ void get_ions_near_channelsMP2D(double* ions, double* buffers, float* buf_free, double* ions_near_channels) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n_total_channels = _n_threads * _n_channels;
	int idx_grid = IndexMP2D(_channel_x[threadIdx.x], _channel_y[threadIdx.x], blockIdx.x);
	for (int j = 0; j < _n_ions; ++j) {
		ions_near_channels[idx + j * n_total_channels] = ions[idx_grid + j * _n_elements];
	}
	for (int j = 0; j < _n_buffers; ++j) {
		ions_near_channels[idx + (j + _n_ions) * n_total_channels] = buffers[idx_grid + j * _n_elements];
	}
	for (int j = 0; j < _n_buffers_unique; ++j) {
		ions_near_channels[idx + (_n_buffers + _n_ions + j) * n_total_channels] = buf_free[idx_grid + j * _n_elements];
	}
}

__device__ __forceinline__ void loadSharedData(float* source, float* sharedDest, int sourceIndex, int sharedIndex, int length) {
	for (int i = 0; i < length; ++i) {
		sharedDest[sharedIndex + i * (blockDim.x + 2) * (blockDim.y + 2) * blockDim.z] = source[sourceIndex + i * _n_elements];
	}
}

__global__ void evolution_operator2DDyad(float* ions, float* buffers, float* evo_ions, float* evo_buffers, float* buf_free, float* currents) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_batch = threadIdx.z + blockDim.z * blockIdx.z;
	//first, create shared memory arrays (including halo cells)
	extern __shared__ float sharedData[];
	float* ions_sh = sharedData;
	float* buffers_sh = &sharedData[_n_ions * (blockDim.x + 2) * (blockDim.y + 2) * blockDim.z];
	if (idx > 0 && idx < _nx - 1 && idy > 0 && idy < _ny - 1 && idx_batch < _n_threads) { //Dirichlet BC
		int n_elem_sh = (blockDim.x + 2) * (blockDim.y + 2) * blockDim.z;
		int idfull = IndexMP2D(idx, idy, idx_batch);
		int idfull_sh = threadIdx.y + 1 + (blockDim.y + 2) * ((threadIdx.x + 1) + (blockDim.x + 2) * threadIdx.z);
		loadSharedData(ions, ions_sh, idfull, idfull_sh, _n_ions);
		loadSharedData(buffers, buffers_sh, idfull, idfull_sh, _n_buffers);
		// Handle boundary cells
		int idfull_temp, idfull_sh_temp;
		if (threadIdx.x == 0 || idx == 1) {
			idfull_temp = IndexMP2D(idx - 1, idy, idx_batch);
			int idx_sh = idx == 1 ? threadIdx.x : 0;
			idfull_sh_temp = threadIdx.y + 1 + (blockDim.y + 2) * (idx_sh + (blockDim.x + 2) * threadIdx.z);
			loadSharedData(ions, ions_sh, idfull_temp, idfull_sh_temp, _n_ions);
			loadSharedData(buffers, buffers_sh, idfull_temp, idfull_sh_temp, _n_buffers);
		}
		if (threadIdx.x == blockDim.x - 1 || idx == _nx - 2) {
			idfull_temp = IndexMP2D(idx + 1, idy, idx_batch);
			int idx_sh = idx == _nx - 2 ? threadIdx.x + 2 : blockDim.x + 1;
			idfull_sh_temp = threadIdx.y + 1 + (blockDim.y + 2) * (idx_sh + (blockDim.x + 2) * threadIdx.z);
			loadSharedData(ions, ions_sh, idfull_temp, idfull_sh_temp, _n_ions);
			loadSharedData(buffers, buffers_sh, idfull_temp, idfull_sh_temp, _n_buffers);
		}
		if (threadIdx.y == 0 || idy == 1) {
			idfull_temp = IndexMP2D(idx, idy - 1, idx_batch);
			int idy_sh = idy == 1 ? threadIdx.y : 0;
			idfull_sh_temp = idy_sh + (blockDim.y + 2) * (threadIdx.x + 1 + (blockDim.x + 2) * threadIdx.z);
			loadSharedData(ions, ions_sh, idfull_temp, idfull_sh_temp, _n_ions);
			loadSharedData(buffers, buffers_sh, idfull_temp, idfull_sh_temp, _n_buffers);
		}
		if (threadIdx.y == blockDim.y - 1 || idy == _ny - 2) {
			idfull_temp = IndexMP2D(idx, idy + 1, idx_batch);
			int idy_sh = idy == _ny - 2 ? threadIdx.y + 2 : blockDim.y + 1;
			idfull_sh_temp = idy_sh + (blockDim.y + 2) * (threadIdx.x + 1 + (blockDim.x + 2) * threadIdx.z);
			loadSharedData(ions, ions_sh, idfull_temp, idfull_sh_temp, _n_ions);
			loadSharedData(buffers, buffers_sh, idfull_temp, idfull_sh_temp, _n_buffers);
		}
		__syncthreads();
		//diffusion part
		int idxl = threadIdx.y + 1 + (blockDim.y + 2) * (threadIdx.x + (blockDim.x + 2) * threadIdx.z);
		int idxr = threadIdx.y + 1 + (blockDim.y + 2) * (threadIdx.x + 2 + (blockDim.x + 2) * threadIdx.z);
		int idyl = threadIdx.y + (blockDim.y + 2) * (threadIdx.x + 1 + (blockDim.x + 2) * threadIdx.z);
		int idyr = threadIdx.y + 2 + (blockDim.y + 2) * (threadIdx.x + 1 + (blockDim.x + 2) * threadIdx.z);
		int str, str_sh;
		for (int j = 0; j < _n_ions; ++j) {
			str_sh = j * n_elem_sh;
			str = j * _n_elements;
			evo_ions[idfull + str] = _D_ions[j] * (d2YdX2MP2D(ions_sh[idxl + str_sh], ions_sh[idfull_sh + str_sh], ions_sh[idxr + str_sh], _dx) +
				d2YdX2MP2D(ions_sh[idyl + str_sh], ions_sh[idfull_sh + str_sh], ions_sh[idyr + str_sh], _dy));
		}
		for (int j = 0; j < _n_buffers; ++j) {
			str_sh = j * n_elem_sh;
			str = j * _n_elements;
			evo_buffers[idfull + str] = _D_buf[j] * (d2YdX2MP2D(buffers_sh[idxl + str_sh], buffers_sh[idfull_sh + str_sh], buffers_sh[idxr + str_sh], _dx) +
				d2YdX2MP2D(buffers_sh[idyl + str_sh], buffers_sh[idfull_sh + str_sh], buffers_sh[idyr + str_sh], _dy));
		}
		//reaction part
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
					str2_sh = idfull_sh + _idx_buff * n_elem_sh;
					str2 = idfull + _idx_buff * _n_elements;
					str3 = idfull + i * _n_elements;
					R1 = _koff[_idx_buff] * buffers_sh[str2_sh] - _kon[_idx_buff] * ions_sh[str_sh] * buf_free[str3];
					evo_buffers[str2] += -R1;
					R += R1;
					++_idx_buff;
				}
			}
			evo_ions[str] += R;
		}
		//source part (current + SL)
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
}

__global__ void evo_stepMP2D(float* evo_ions_total, double* ions, float* ions_f, float* evo_buffers_total, double* buffers, float* buffers_f) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_batch = threadIdx.z + blockDim.z * blockIdx.z;
	int idfull = IndexMP2D(idx, idy, idx_batch);
	double tot_double;
	if (idx < _nx && idy < _ny &&  idx_batch < _n_threads) {
		if (!((idx == 0 || idx == _nx - 1) || (idy == 0 || idy == _ny - 1))) {
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
}

__global__ void get_free_buffersMP2D(float* buffers, float* buf_free) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_batch = threadIdx.z + blockDim.z * blockIdx.z;
	int idfull = IndexMP2D(idx, idy, idx_batch);
	if (idx < _nx && idy < _ny && idx_batch < _n_threads) {
		if (!((idx == 0 || idx == _nx - 1) || (idy == 0 || idy == _ny - 1))) {
			int _idx_buff = 0;
			for (int j = 0; j < _n_ions; ++j) {
				int idx2 = idfull + j * _n_elements;
				int idx3;
				if (j == 0) {
					for (int i = 0; i < _n_buffers_unique; ++i)
						buf_free[idfull + i * _n_elements] = _buf_tot[i];
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


//Gradients dimensions: x - x coordinate, block - scenario. Mixed precision part (calculates in double precision)
__global__ void gradient_xMP2D(double* ions, double* buffers, double* grads_x) {
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
		idy = 1;
		idfull = IndexMP2D(idx, idy, n_batch);
		idy = 0;
		idfull2 = IndexMP2D(idx, idy, n_batch);
		idy = 2;
		idfull3 = IndexMP2D(idx, idy, n_batch);
		for (int j = 0; j < _n_ions; ++j) {
			str = j * _n_elements;
			gradients_x[idx + blocksize * j] += (2 * ions[idfull + str] - 1.5 * ions[idfull2 + str] - 0.5 * ions[idfull3 + str]);
		}
		for (int i = 0; i < _n_buffers; ++i) {
			str = i * _n_elements;
			gradients_x[idx + blocksize * (_n_ions + i)] += (2 * buffers[idfull + str] - 1.5 * buffers[idfull2 + str] - 0.5 * buffers[idfull3 + str]);
		}
		idy = _ny - 2;
		idfull = IndexMP2D(idx, idy, n_batch);
		idy = _ny - 1;
		idfull2 = IndexMP2D(idx, idy, n_batch);
		idy = _ny - 3;
		idfull3 = IndexMP2D(idx, idy, n_batch);
		for (int j = 0; j < _n_ions; ++j) {
			str = j * _n_elements;
			gradients_x[idx + blocksize * j] += (2 * ions[idfull + str] - 1.5 * ions[idfull2 + str] - 0.5 * ions[idfull3 + str]);
		}
		for (int i = 0; i < _n_buffers; ++i) {
			str = i * _n_elements;
			gradients_x[idx + blocksize * (_n_ions + i)] += (2 * buffers[idfull + str] - 1.5 * buffers[idfull2 + str] - 0.5 * buffers[idfull3 + str]);
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
			grads_x[j + _n_ions_buffers * n_batch] = _D_ions[j] * _grad_mult * gradients_x[blocksize * j] * _sx;
		for (int i = 0; i < _n_buffers; ++i)
			grads_x[i + _n_ions + _n_ions_buffers * n_batch] = _D_buf[i] * _grad_mult * gradients_x[blocksize * (_n_ions + i)] * _sx;
	}
}

__global__ void gradient_yMP2D(double* ions, double* buffers, double* grads_x) {
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
		idx = 1;
		idfull = IndexMP2D(idx, idy, n_batch);
		idx = 0;
		idfull2 = IndexMP2D(idx, idy, n_batch);
		idx = 2;
		idfull3 = IndexMP2D(idx, idy, n_batch);
		for (int j = 0; j < _n_ions; ++j) {
			str = j * _n_elements;
			gradients_x[threadIdx.x + blocksize * j] += (2 * ions[idfull + str] - 1.5 * ions[idfull2 + str] - 0.5 * ions[idfull3 + str]);
		}
		for (int i = 0; i < _n_buffers; ++i) {
			str = i * _n_elements;
			gradients_x[threadIdx.x + blocksize * (_n_ions + i)] += (2 * buffers[idfull + str] - 1.5 * buffers[idfull2 + str] - 0.5 * buffers[idfull3 + str]);
		}
		idx = _nx - 2;
		idfull = IndexMP2D(idx, idy, n_batch);
		idx = _nx - 1;
		idfull2 = IndexMP2D(idx, idy, n_batch);
		idx = _nx - 3;
		idfull3 = IndexMP2D(idx, idy, n_batch);
		for (int j = 0; j < _n_ions; ++j) {
			str = j * _n_elements;
			gradients_x[threadIdx.x + blocksize * j] += (2 * ions[idfull + str] - 1.5 * ions[idfull2 + str] - 0.5 * ions[idfull3 + str]);
		}
		for (int i = 0; i < _n_buffers; ++i) {
			str = i * _n_elements;
			gradients_x[threadIdx.x + blocksize * (_n_ions + i)] += (2 * buffers[idfull + str] - 1.5 * buffers[idfull2 + str] - 0.5 * buffers[idfull3 + str]);
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
			grads_x[j + _n_ions_buffers * n_batch] = _D_ions[j] * _grad_mult * gradients_x[blocksize * j] * _sy;
		for (int i = 0; i < _n_buffers; ++i)
			grads_x[i + _n_ions + _n_ions_buffers * n_batch] = _D_buf[i] * _grad_mult * gradients_x[blocksize * (_n_ions + i)] * _sy;
	}
}

__global__ void gradient_sumMP2D(double* grads_x, double* grads_y, double* grads) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	grads[idx] = grads_x[idx] + grads_y[idx];
}

__global__ void set_boundariesMP2D(float* ions, float* buffers, double* ions_d, double* buffers_d, double* ions_boundary, double* buffers_boundary) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_batch = threadIdx.z + blockDim.z * blockIdx.z;
	int idfull = IndexMP2D(idx, idy, idx_batch);
	if (idx < _nx && idy < _ny && idx_batch < _n_threads) {
		if (((idx == 0 || idx == _nx - 1) || (idy == 0 || idy == _ny - 1))) {
			int idx2;
			for (int j = 0; j < _n_ions; ++j) {
				idx2 = idfull + j * _n_elements;
				ions[idx2] = ions_boundary[idx_batch + j * _n_threads];
				ions_d[idx2] = ions_boundary[idx_batch + j * _n_threads];
			}
			for (int i = 0; i < _n_buffers; ++i) {
				idx2 = idfull + i * _n_elements;
				buffers[idx2] = buffers_boundary[idx_batch + i * _n_threads];
				buffers_d[idx2] = buffers_boundary[idx_batch + i * _n_threads];
			}
		}
	}
}


DyadRDMP2D::DyadRDMP2D(nlohmann::json& j, nlohmann::json& j_jsr, int nthreads) {
	dx = j["dx"];
	cudaMemcpyToSymbol(_dx, &dx, sizeof(float));
	dy = j["dy"];
	cudaMemcpyToSymbol(_dy, &dy, sizeof(float));
	x = j["x"];
	y = j["y"];
	z = j["z"];
	blockx = j["CUDA"]["BLOCK X"];
	blocky = j["CUDA"]["BLOCK Y"];
	blockz = j["CUDA"]["BLOCK Z"];
	R = j["Radius"];
	V = j["Voltage"];
	block.x = blockx, block.y = blocky, block.z = blockz;
	n_threads = nthreads;
	cudaStreamCreate(&(stream_dyad_mp2d));
	cudaStreamCreate(&(stream_dyad_2_mp2d));
	cudaEventCreate(&grad_event);
	cudaEventCreate(&grad_event2);
	cudaMemcpyToSymbol(_n_threads, &n_threads, sizeof(n_threads));
	nx = x / dx + 1;
	cudaMemcpyToSymbol(_nx, &nx, sizeof(nx));
	ny = y / dy + 1;
	cudaMemcpyToSymbol(_ny, &ny, sizeof(ny));
	dyad_dims.resize(4);
	dyad_dims[1] = n_threads;
	dyad_dims[2] = nx;
	dyad_dims[3] = ny;
	double sx = dy * z / dx;
	double sy = dx * z / dy;
	cudaMemcpyToSymbol(_sx, &sx, sizeof(sx));
	cudaMemcpyToSymbol(_sy, &sy, sizeof(sy));
	int n_elements_thread = nx * ny;
	cudaMemcpyToSymbol(_n_elements_per_thread, &n_elements_thread, sizeof(n_elements_thread));
	grid.x = (nx + block.x - 1) / blockx, grid.y = (ny + block.y - 1) / blocky, grid.z = (n_threads + block.z - 1) / blockz;
	n_elements = nx * ny * n_threads;
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
		ions.push_back(std::make_unique<Ion<float> >(el.key(), pars_ion["D"], pars_ion["Concentration"]));
		if (el.key() == "Calcium")
			ca_idx = n_ions;
		++n_ions;
		D_ions.push_back(pars_ion["D"]);
	}
	total_sr_current.resize(n_ions * n_threads);
	//Buffers
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
	channels = std::make_unique<DyadChannels>(j, j_jsr, nthreads, ions, buffers, V);
	//Channels
	sl_size = channels->GetNSLIonChannels();
	sr_size = channels->GetNSRIonChannels();
	n_channels = sl_size + sr_size;
	for (int j = 0; j < sr_size; ++j) {
		auto p = channels->GetSRIonChannel(0, j)->GetCoordinates()->GetCoordsOnGrid(dx, dy, 1e-5);
		x_idx_v.push_back(p[0]);
		y_idx_v.push_back(p[1]);
		z_idx_v.push_back(p[2]);
	}
	for (int j = sr_size; j < n_channels; ++j) {
		auto p = channels->GetSLIonChannel(0, j - sr_size)->GetCoordinates()->GetCoordsOnGrid(dx, dy, 1e-5);
		x_idx_v.push_back(p[0]);
		y_idx_v.push_back(p[1]);
		z_idx_v.push_back(p[2]);
	}
	cudaMemcpyToSymbol(_n_channels, &n_channels, sizeof(n_channels));
	n_buffers = idx_buf;
	int n_buf_unique = buffers.size();
	n_elements_near_channels = (n_ions + n_buffers + n_buf_unique) * n_channels * n_threads;
	currents.resize(n_channels * n_ions * n_threads);
	channels_ions_dims.resize(3);
	channels_ions_dims[1] = n_threads;
	channels_ions_dims[2] = n_channels;
	channels_dims.resize(3);
	channels_dims[1] = n_threads;
	channels_dims[2] = n_channels;
	cudaMalloc(&d_currents, n_channels * n_ions * n_threads * sizeof(float));
	cudaMemcpyToSymbol(_channel_x, x_idx_v.data(), sizeof(int) * x_idx_v.size());
	cudaMemcpyToSymbol(_channel_y, y_idx_v.data(), sizeof(int) * y_idx_v.size());
	ions_near_channels.resize(n_elements_near_channels);
	cudaMemcpyToSymbol(_n_ions, &n_ions, sizeof(n_ions));
	for (auto const& el : j["Sarcolemma"].items()) {
		auto pars_SL = el.value();
		ions_and_SL[el.key()] = std::make_unique<IonSarcolemma<float> >(pars_SL["N1"], pars_SL["K1"], pars_SL["N2"], pars_SL["K2"]);
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
	cudaMemcpyToSymbol(_N1, N1.data(), sizeof(float) * N1.size());
	cudaMemcpyToSymbol(_N2, N2.data(), sizeof(float) * N2.size());
	cudaMemcpyToSymbol(_ion_SL_table, SL_binds_ion.data(), sizeof(short) * SL_binds_ion.size());
	//calculate multiplier for gradient
	pointers_are_set = false;
	dt_is_set = false;
	grad_mult = 3.0f / (4.0 * acos(-1) * R * R * R); /// (0.566233 * R * R * R); //
	cudaMemcpyToSymbol(_grad_mult, &grad_mult, sizeof(grad_mult));
	cudaMemcpyToSymbol(_n_buffers, &n_buffers, sizeof(n_buffers));
	cudaMemcpyToSymbol(_n_buffers_unique, &n_buf_unique, sizeof(n_buffers));
	n_blocks_init = (n_elements + threads_per_block - 1) / threads_per_block;
	int n_ions_and_buffers = idx_buf + n_ions;
	cudaMemcpyToSymbol(_n_ions_buffers, &n_ions_and_buffers, sizeof(n_ions_and_buffers));
	cudaMalloc(&d_gradients_x, (idx_buf + n_ions) * n_threads * sizeof(double));
	cudaMalloc(&d_gradients_y, (idx_buf + n_ions) * n_threads * sizeof(double));
	cudaMalloc(&d_gradients, (idx_buf + n_ions) * n_threads * sizeof(double));
	gradients = new double[(idx_buf + n_ions) * n_threads];
	cudaMalloc(&d_ions, n_elements * n_ions * sizeof(double));
	cudaMalloc(&d_ions_f, n_elements * n_ions * sizeof(float));
	cudaMalloc(&evo_ions_total, n_elements * n_ions * sizeof(float));
	cudaMalloc(&evo_buffers_total, n_elements * idx_buf * sizeof(float)); 
	cudaMalloc(&d_ions_near_channels, n_elements_near_channels * sizeof(double));
	std::vector<float> ions_b;
	for (auto const& i : ions)
		ions_b.push_back(i->Cb);
	cudaMemcpyToSymbol(_ionsb, ions_b.data(), sizeof(float)* ions_b.size());
	cudaMemcpyToSymbol(_D_ions, D_ions.data(), sizeof(float)* D_ions.size());
	cudaMemcpyToSymbol(_D_buf, D_buf.data(), sizeof(float)* D_buf.size());
	cudaMemcpyToSymbol(_kon, kon_buf.data(), sizeof(float)* kon_buf.size());
	cudaMemcpyToSymbol(_koff, koff_buf.data(), sizeof(float)* koff_buf.size());
	cudaMemcpyToSymbol(_buf_tot, total_buf.data(), sizeof(float)* total_buf.size());
	cudaMemcpyToSymbol(_ion_buffer_table, buffer_binds_ion.data(), sizeof(short)* buffer_binds_ion.size());
	cudaMemcpyToSymbol(_ion_SL_table, SL_binds_ion.data(), sizeof(short)* SL_binds_ion.size());
	cudaMalloc(&d_buffers, n_elements * idx_buf * sizeof(double));
	cudaMalloc(&d_buffers_f, n_elements * idx_buf * sizeof(float));
	cudaMalloc(&d_buffers_free, n_elements * n_buf_unique * sizeof(float));
	cudaMalloc(&d_currents_grid, n_elements * n_ions * sizeof(float));
	cudaMalloc(&d_init_buffers, idx_buf * sizeof(float));
	cudaMemcpy(d_init_buffers, buf_init.data(), idx_buf * sizeof(float), cudaMemcpyHostToDevice);
}

void DyadRDMP2D::_ZeroCurrents() {
	for (int i = 0; i < n_threads; ++i)
		for (int j = 0; j < n_channels; ++j)
			for (int k = 0; k < n_ions; ++k)
				ions_near_channels[j + n_channels * (i + k * n_threads)] = 0;
	for (int i = 0; i < n_threads; ++i)
		for (int j = 0; j < n_channels; ++j)
			for (int k = 0; k < n_buffers; ++k)
				ions_near_channels[j + n_channels * (i + (k +n_ions) * n_threads)] = 0;
}

void DyadRDMP2D::_SetupUpdateGraph() {
	cudaStreamBeginCapture(stream_dyad_mp2d, cudaStreamCaptureModeGlobal);
	if (is_cytosol_cpu) {
		cudaMemcpyAsync(d_ions_from_cytosol, ions_from_cytosol, n_ions * n_threads * sizeof(double), cudaMemcpyHostToDevice, stream_dyad_mp2d);
		cudaMemcpyAsync(d_buffers_from_cytosol, buffers_from_cytosol, idx_buf * n_threads * sizeof(double), cudaMemcpyHostToDevice, stream_dyad_mp2d);
	}
	get_ions_near_channelsMP2D << <n_threads, n_channels, 0, stream_dyad_mp2d >> > (d_ions, d_buffers, d_buffers_free, d_ions_near_channels);
	cudaMemcpyAsync(ions_near_channels.data(), d_ions_near_channels, n_elements_near_channels * sizeof(double), cudaMemcpyDeviceToHost, stream_dyad_mp2d);
	set_boundariesMP2D << <grid, block, 0, stream_dyad_mp2d >> > (d_ions_f, d_buffers_f, d_ions, d_buffers, d_ions_from_cytosol, d_buffers_from_cytosol);
	gradient_xMP2D << <n_threads, n_threads_grad_x, n_threads_grad_x* (idx_buf + n_ions) * sizeof(double), stream_dyad_mp2d >> > (d_ions, d_buffers, d_gradients_x);
	gradient_yMP2D << <n_threads, n_threads_grad_y, n_threads_grad_y* (idx_buf + n_ions) * sizeof(double), stream_dyad_mp2d >> > (d_ions, d_buffers, d_gradients_y);
	gradient_sumMP2D << <1, n_threads* (idx_buf + n_ions), 0, stream_dyad_mp2d >> > (d_gradients_x, d_gradients_y, d_gradients);
	if (is_cytosol_cpu)
		cudaMemcpyAsync(gradients, d_gradients, (idx_buf + n_ions) * n_threads * sizeof(double), cudaMemcpyDeviceToHost, stream_dyad_mp2d);
	cudaStreamEndCapture(stream_dyad_mp2d, &update_graph_mp2d);
	cudaGraphInstantiate(&update_instance_mp2d, update_graph_mp2d, NULL, NULL, 0);
}

void DyadRDMP2D::_SetupStepGraph() {
	cudaStreamBeginCapture(stream_dyad_mp2d, cudaStreamCaptureModeGlobal);
	evolution_operator2DDyad << <grid, block, (n_ions + n_buffers)* (block.x + 2)* (block.y + 2)* block.z * sizeof(float), stream_dyad_mp2d >> > (d_ions_f, d_buffers_f, evo_ions_total, evo_buffers_total, d_buffers_free, d_currents_grid);
	evo_stepMP2D << <grid, block, 0, stream_dyad_mp2d >> > (evo_ions_total, d_ions, d_ions_f, evo_buffers_total, d_buffers, d_buffers_f); //DOUBLE PRECISION KERNEL 
	get_free_buffersMP2D << <grid, block, 0, stream_dyad_mp2d >> > (d_buffers_f, d_buffers_free);
	cudaStreamEndCapture(stream_dyad_mp2d, &timestep_graph_mp2d);
	cudaGraphInstantiate(&timestep_instance_mp2d, timestep_graph_mp2d, NULL, NULL, 0);
}

std::vector<double> DyadRDMP2D::GaussElimination(std::vector<std::vector<double> >& a) {
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

double DyadRDMP2D::GetElementVolume() {
	return dx * dy * z;
}

int DyadRDMP2D::GetNumSRChannels() {
	return sr_size;
}
void DyadRDMP2D::Reset() {
	set_initialsMP2D << <grid, block >> > (d_ions_f, d_buffers_f, d_init_buffers, d_currents_grid, d_buffers_free, d_gradients, d_ions, d_buffers);
	_ZeroCurrents();
}

bool DyadRDMP2D::UsesGPU() {
	return true;
}

int DyadRDMP2D::GetNIons() {
	return n_ions;
}

std::vector<double> DyadRDMP2D::GetTotalSRCurrent() {
	return total_sr_current;
}

void DyadRDMP2D::InitOpening(int thread, int channel) {
	channels->InitOpening(thread, channel);
}

void DyadRDMP2D::_SetCurrents(const std::vector<double>& ions_jsr, const std::vector<double>& ions_extracell) {
	jsr_ions = &ions_jsr;
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
	cudaMemcpyAsync(d_currents, currents.data(), currents.size() * sizeof(float), cudaMemcpyHostToDevice, stream_dyad_mp2d);
	set_currentsMP2D << <n_threads, n_channels, 0, stream_dyad_mp2d >> > (d_currents_grid, d_currents);
}

void DyadRDMP2D::GetEffluxes(double*& currents_out) {
	if (is_cytosol_cpu) {
		cudaDeviceSynchronize();
		currents_out = gradients;
		return;
	}
	currents_out = d_gradients;
}

void DyadRDMP2D::Update(double*& ions_cytosol, double*& buffers_cytosol, const std::vector<double>& ions_jsr, const std::vector<double>& extracellular_ions, double &Vcyt) {
	V = Vcyt;
	if (ions_from_cytosol != ions_cytosol || buffers_from_cytosol != buffers_cytosol) {
		ions_from_cytosol = ions_cytosol;
		buffers_from_cytosol = buffers_cytosol;
		if (is_cytosol_cpu) {
			cudaMalloc(&d_ions_from_cytosol, n_ions * n_threads * sizeof(double));
			cudaMalloc(&d_buffers_from_cytosol, idx_buf  * n_threads * sizeof(double));
		}
		else {
			d_ions_from_cytosol = ions_cytosol;
			d_buffers_from_cytosol = buffers_cytosol;
		}
		_SetupStepGraph();
		_SetupUpdateGraph();
	}
	cudaGraphLaunch(update_instance_mp2d, stream_dyad_mp2d);
	cudaDeviceSynchronize();
	_SetCurrents(ions_jsr, extracellular_ions);
}

//Iteration on calcium and buffers concentrations in dyad 
void DyadRDMP2D::RunRD(double dt, int idx_b) {
	double dtt = dt;
	if (!dt_is_set) {
		cudaMemcpyToSymbol(_dt, &dtt, sizeof(double));
		dt_is_set = true;
	}
	cudaGraphLaunch(timestep_instance_mp2d, stream_dyad_mp2d);
}

void DyadRDMP2D::RunMC(double dt, int n_thread) {
	channels->RunMC(dt, n_thread, ions_near_channels,*jsr_ions,V);
}

double DyadRDMP2D::GetL() {
	return R;
}

std::vector<std::string> DyadRDMP2D::GetListofIons() {
	std::vector<std::string> out;
	for (auto &ion : ions)
		out.push_back(ion->name);
	return out;
}

std::map < std::string, std::vector<double> > DyadRDMP2D::GetConcentrations(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions.size(); ++i) {
		auto it = std::find(values.begin(), values.end(), ions[i]->name);
		if (it != values.end()) {
			out.insert(std::pair<std::string, std::vector<double> >(ions[i]->name, std::vector<double>(n_elements)));
			cudaMemcpyAsync(out[ions[i]->name].data(), d_ions + i * n_elements, n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream_dyad_mp2d);
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
					cudaMemcpyAsync(out[type_name].data(), d_buffers + idx_b * n_elements, n_elements * sizeof(double), cudaMemcpyDeviceToHost, stream_dyad_mp2d);
				}
			}
		}
	}
	cudaDeviceSynchronize();
	return out;
}

std::map < std::string, std::vector<int> > DyadRDMP2D::GetChannelsStates(std::vector<std::string>& values) {
	std::map < std::string, std::vector<int> > out;
	std::string name = "RyR";
	std::vector<int> states;
	for (int i = 0; i < n_threads; ++i)
		for (int j = 0; j < n_channels; ++j)
			states.push_back(channels->GetSRIonChannel(i, j)->GetKineticModel()->GetState());
	out[name] = states;
	return out;
}

std::map <std::string, std::vector<double> > DyadRDMP2D::GetIonsNearChannels(std::vector<std::string>& values) {
	std::map < std::string, std::vector<double> > out;
	for (int i = 0; i < ions.size(); ++i) {
		auto itb = ions_near_channels.begin() + i * n_threads * n_channels;
		auto ite = ions_near_channels.begin() + (i + 1) * n_threads * n_channels;
		out.insert(std::pair<std::string, std::vector<double> >(ions[i]->name, std::vector<double>(itb, ite)));
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
					int idx_b = idx_bufion[buf_name][ion_name];
					auto itb = ions_near_channels.begin() + (idx_b + n_ions) * n_threads * n_channels;
					auto ite = ions_near_channels.begin() + (idx_b + n_ions + 1) * n_threads * n_channels;
					out.insert(std::pair<std::string, std::vector<double> >(type_name, std::vector<double>(itb, ite)));
				}
			}
		}
	}
	cudaDeviceSynchronize();
	return out;
}

DyadRDMP2D::~DyadRDMP2D() {
	delete[] gradients;
	cudaFree(&d_ions);
	cudaFree(&d_ions_f);
	cudaFree(&d_ions_near_channels);
	cudaFree(&evo_ions_total);
	cudaFree(&d_buffers);
	cudaFree(&d_buffers_f);
	cudaFree(&d_buffers_free);
	cudaFree(&d_init_buffers);
	cudaFree(&d_gradients_x);
	cudaFree(&d_gradients_y);
	cudaFree(&d_gradients);
	if (is_cytosol_cpu) {
		cudaFree(&d_ions_from_cytosol);
		cudaFree(&d_buffers_from_cytosol);
	}
	cudaFree(&evo_buffers_total);
	cudaFree(&d_currents);
	cudaFree(&d_currents_grid);
	cudaGraphExecDestroy(timestep_instance_mp2d);
	cudaGraphExecDestroy(update_instance_mp2d);
	cudaGraphDestroy(timestep_graph_mp2d);
	cudaGraphDestroy(update_graph_mp2d);
	cudaStreamDestroy(stream_dyad_mp2d);
	cudaStreamDestroy(stream_dyad_2_mp2d);
	cudaEventDestroy(grad_event);
	cudaEventDestroy(grad_event2);
}
