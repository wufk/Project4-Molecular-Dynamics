#include "md.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sm_20_atomic_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <thrust/random.h>

#include "device_functions.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

#include "memory.h"
#include "random.h"

float m = 1; float inv_m = 1. / m;

float dt = 0.0005;
float hdt = 0.5 * dt;

float sigma = 1, epsilon = 1, A, B, C, D;
float dcut = 2.5 * sigma;
float dcut2 = dcut * dcut;

float t = 0.0;

float tau = 0.01 / dt;

float kB = 1., ke = 0.0, pe = 0.0;
float T;
float T0 = 0.5, Tt = 10.5;

int Crate = 20;

int ifreq = 10, nstep = 100;
int N = (Tt - T0) / (Crate * dt * nstep);

int nunit = 4;
int nall = 4;

char Atom[3] = "Cr";

Memory* M = new Memory();;
RanPark* rnd = new RanPark(1234);;

float ncell[3] = { 4.0f, 4.0f, 4.0f };
float a0[3] = { 1.5f, 1.5f, 1.5f };
glm::vec3 L(ncell[0] * a0[0], ncell[1] * a0[1], ncell[2] * a0[2]);
glm::vec3 hL(L[0] / 2, L[1] / 2, L[2] / 2);

glm::vec3 *dev_vel = nullptr;
glm::vec3 *dev_pos = nullptr;
glm::vec3 *dev_force = nullptr;
glm::vec3 *h_vel = nullptr;
glm::vec3 *h_pos = nullptr;
glm::vec3 *h_force = nullptr;
float *ke_idata = nullptr;
float *ke_odata = nullptr;
float *dev_pe = nullptr;
const int threads = 256;

thrust::minstd_rand rng;
thrust::uniform_real_distribution<float> unitDistrib(-0.5, 0.5);

cudaEvent_t start;
cudaEvent_t end;

curandState_t* states = nullptr;

////////////////////////////////////////////// Velocity //////////////////////////////////////////////
__global__ void kernInitVel(int n, glm::vec3 *vel, curandState_t* states)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) return;

	float vx = curand_uniform(&(states[tid])) - 0.5f;
	float vy = curand_uniform(&(states[tid])) - 0.5f;
	float vz = curand_uniform(&(states[tid])) - 0.5f;
	vel[tid] = glm::vec3(vx,vy,vz);
}

__global__ void kernVelMinus(int n, glm::vec3 *vel, glm::vec3 mon) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) return;

	vel[tid] -= mon;
}

__global__ void kernVelMultiply(int n, glm::vec3 *vel, float gama) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) return;

	vel[tid] *= gama;
}

__global__ void kernComputeDotProduct(int n, glm::vec3 *vel, float *d_odata)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) return;

	d_odata[tid] = glm::dot(vel[tid], vel[tid]);
}

//////////////////////////////////// reduce energy //////////////////////////////////////////////////////////
__global__ void reduce_energy(const float* d_idata, float* d_odata, int n)
{
	extern __shared__ float shm[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n)
		shm[threadIdx.x] = d_idata[idx];

	__syncthreads();

	for (int c = blockDim.x / 2; c > 0; c >>= 1)
	{

		if (threadIdx.x < c)
			shm[threadIdx.x] += shm[threadIdx.x + c];

		__syncthreads();
	}

	if (threadIdx.x == 0)
		d_odata[blockIdx.x] = shm[0];
}

float reduce_energy_wrapper(const float *d_idata, float *d_odata, const int elements)
{
	int dimThreads = threads;
	int dimBlocks = (elements + dimThreads - 1) / (dimThreads);

	if (elements < dimThreads) {
		float *h_blocks = (float *)malloc(elements * sizeof(float));
		cudaMemcpy(h_blocks, d_odata, elements * sizeof(float), cudaMemcpyDeviceToHost);

		float gpu_result = 0;

		for (int i = 0; i < elements; i++)
			gpu_result += h_blocks[i];

		free(h_blocks);

		return gpu_result;
	}
	else {
		reduce_energy << <dimBlocks, dimThreads, sizeof(float) * dimThreads >> >(d_idata, d_odata, elements);

		return reduce_energy_wrapper(d_odata, d_odata, dimBlocks);
	}
}

/////////////////////////////////////////// Reduce velocity //////////////////////////////////////////////
__global__ void vel_reduce(const glm::vec3* d_idata, glm::vec3* d_odata, int n)
{
	extern __shared__ glm::vec3 smem[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n)
		smem[threadIdx.x] = d_idata[idx];

	__syncthreads();

	for (int c = blockDim.x / 2; c > 0; c >>= 1)
	{
		if (threadIdx.x < c)
			smem[threadIdx.x] += smem[threadIdx.x + c];

		__syncthreads();
	}

	if (threadIdx.x == 0)
		d_odata[blockIdx.x] = smem[0];
}

glm::vec3 vel_reduce_wrapper(const glm::vec3 *d_idata, glm::vec3 *d_odata, const int elements)
{
	int dimThreads = threads;
	int dimBlocks = (elements + dimThreads - 1) / (dimThreads);

	if (elements < dimThreads) {
		glm::vec3 *h_blocks = (glm::vec3 *)malloc(elements * sizeof(glm::vec3));
		cudaMemcpy(h_blocks, d_odata, elements * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

		glm::vec3 gpu_result = glm::vec3(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < elements; i++)
			gpu_result += h_blocks[i];

		free(h_blocks);

		return gpu_result;
	}
	else {
		vel_reduce << <dimBlocks, dimThreads, sizeof(glm::vec3) * dimThreads >> >(d_idata, d_odata, elements);

		return vel_reduce_wrapper(d_odata, d_odata, dimBlocks);
	}
}

////////////////////////////////////////////////////
__global__ void kernNaiveVelocityIntegration(int n, glm::vec3 *vel, glm::vec3 *force, float inv_m, float hdt)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) return;
	vel[tid] += force[tid] * inv_m * hdt;
}

__global__ void kernNaivePositionIntegration(int n, glm::vec3 *pos, glm::vec3 *vel, float dt)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) return;
	pos[tid] += dt * vel[tid];
}

__global__ void kernNaiveForce(int n, glm::vec3 *pos, glm::vec3 *force, glm::vec3 *vel, float coef, glm::vec3 hL, glm::vec3 L,
	float dcut2, float A, float B, float C, float D, float m, float tau, float *pe, curandState_t* states)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) return;

	float wx = curand_uniform(&(states[tid])) - 0.5f;
	float wy = curand_uniform(&(states[tid])) - 0.5f;
	float wz = curand_uniform(&(states[tid])) - 0.5f;
	glm::vec3 w = glm::vec3(wx,wy,wz);

	for (int i = tid + 1; i < n; i++) {

		glm::vec3 dx = pos[i] - pos[tid];
		while (dx.x > hL.x) dx.x -= L.x;
		while (dx.y > hL.y) dx.y -= L.y;
		while (dx.z > hL.z) dx.z -= L.z;
		while (-dx.x > hL.x) dx.x += L.x;
		while (-dx.y > hL.y) dx.y += L.y;
		while (-dx.z > hL.z) dx.z += L.z;

		float r2 = glm::dot(dx, dx);
		float r6 = r2 * r2 *r2;
		float r12 = r6 * r6;

		if (r2 < dcut2) {
			dx *= (A * 1. / r12 + B * 1. / r6) / r2;
			atomicAdd(&(force[tid].x), -dx.x);
			atomicAdd(&(force[tid].y), -dx.y);
			atomicAdd(&(force[tid].z), -dx.z);
			atomicAdd(&(force[i].x), dx.x);
			atomicAdd(&(force[i].y), dx.y);
			atomicAdd(&(force[i].z), dx.z);
			atomicAdd(pe, C * 1. / r12 + D * 1. / r6);
		}
	}
	__syncthreads();
	force[tid] += -m * tau * vel[tid] + coef * w;
}

void forceCPU(int k) {
	float TT = T0;
	float Gc = (Tt - T0) / (N - 1);
	TT = TT + k * Gc;
	//printf("TT: %f\n", TT);
	pe = 0.0;
	float coef = sqrt(24. * tau * m * kB * TT / dt);//coef to calculate w
	float dcut = 2.5 * sigma;
	float dcut2 = dcut * dcut;

	for (int i = 0; i < nall; i++){
		h_force[i] = glm::vec3(0.0f, 0.0f, 0.0f);
	}

	for (int i = 0; i < nall - 1; i++) {
		for (int j = i + 1; j < nall; j++) {
			glm::vec3 dx = h_pos[j] - h_pos[i];
			for (int k = 0; k < 3; k++) {
				while (dx.x > hL.x) dx.x -= L.x;
				while (dx.y > hL.y) dx.y -= L.y;
				while (dx.z > hL.z) dx.z -= L.z;
				while (dx.x < -hL.x) dx.x += L.x;
				while (dx.y < -hL.y) dx.y += L.y;
				while (dx.z < -hL.z) dx.z += L.z;
			}
			float r2 = glm::dot(dx, dx);

			float r6 = r2 * r2 *r2;
			float r12 = r6 * r6;

			if (r2 < dcut2) {
				dx *= (A * 1. / r12 + B * 1. / r6) / r2;
				h_force[i] -= dx;
				h_force[j] += dx;
				pe += C * 1. / r12 + D * 1. / r6;
			}
		}
	}

	for (int i = 0; i < nall; i++) {
		float a = (float)unitDistrib(rng);
		float b = (float)unitDistrib(rng);
		float c = (float)unitDistrib(rng);
		glm::vec3 w(a, b, c);
		w *= coef;
		h_force[i] += -m * tau * h_vel[i] + w;
	}

}

__global__ void kernInitRandom(int n, unsigned int seed, curandState_t* states)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) return;
	curand_init(seed, tid, 0, &states[tid]);
}

void MD::MD_init()
{
	//init variables
	float sigma3 = sigma  * sigma * sigma;
	float sigma6 = sigma3 * sigma3;
	float sigma12 = sigma6 * sigma6;
	A = 48. * epsilon * sigma12;
	B = -24. * epsilon * sigma6;
	C = 4. * epsilon * sigma12;
	D = -4. * epsilon * sigma6;

	for (int i = 0; i < 3; i++) nall *= ncell[i];

	int dimThreads = threads;
	int dimBlocks = (nall + dimThreads - 1) / (dimThreads);

	cudaMalloc((void**)&dev_pe, sizeof(float));
	cudaMalloc((void**)&dev_pos, nall * sizeof(glm::vec3));
	cudaMalloc((void**)&dev_vel, nall * sizeof(glm::vec3));
	cudaMalloc((void**)&dev_force, nall * sizeof(glm::vec3));
	cudaMalloc((void**)&ke_idata, nall * sizeof(glm::vec3));
	cudaMalloc((void**)&ke_odata, dimBlocks * sizeof(glm::vec3));
	cudaMalloc((void**)&states, nall * sizeof(curandState_t));

	kernInitRandom << <dimBlocks, dimThreads >> > (nall, 0, states);


	h_pos = new glm::vec3[nall];
	h_vel = new glm::vec3[nall];
	h_force = new glm::vec3[nall];

	//init cell and pos
	float **cell = nullptr;
	M->create(cell, 4, 3, "cell");
	cell[0][0] = cell[0][1] = cell[0][2] = 0.0;
	cell[1][0] = 0.0; cell[1][1] = 0.5 * a0[1]; cell[1][2] = 0.5 * a0[2];
	cell[2][0] = 0.5 * a0[0]; cell[2][1] = 0.0; cell[2][2] = 0.5 * a0[2];
	cell[3][0] = 0.5 * a0[0]; cell[3][1] = 0.5 * a0[1]; cell[3][2] = 0.0;
	int ii = 0;
	for (int ix = 0; ix < ncell[0]; ix++) {
		for (int iy = 0; iy < ncell[1]; iy++) {
			for (int iz = 0; iz < ncell[2]; iz++) {
				for (int iu = 0; iu < nunit; iu++) {
					h_pos[ii].x = float(ix) * a0[0] + cell[iu][0];
					h_pos[ii].y = float(iy) * a0[1] + cell[iu][1];
					h_pos[ii].z = float(iz) * a0[2] + cell[iu][2];
					++ii;
				}
			}
		}
	}
	cudaMemcpy(dev_pos, h_pos, nall * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	M->destroy(cell);

	//////////////////////////////
	kernInitVel << <dimBlocks, dimThreads >> > (nall, dev_vel, states);
	glm::vec3 *vel_odata = nullptr;
	cudaMalloc((void**)&vel_odata, dimBlocks * sizeof(glm::vec3));
	glm::vec3 mon = vel_reduce_wrapper(dev_vel, vel_odata, nall);
	cudaFree(vel_odata);
	mon /= float(nall);
	kernVelMinus << <dimBlocks, dimThreads >> >(nall, dev_vel, mon);

	kernComputeDotProduct << <dimBlocks, dimThreads >> > (nall, dev_vel, ke_idata);
	ke = reduce_energy_wrapper(ke_idata, ke_odata, nall);

	ke *= 0.5 * m; T = ke / (1.5 * float(nall) * kB);
	float gamma = sqrt(T0 / T);

	ke = 0.0;
	kernVelMultiply << <dimBlocks, dimThreads >> > (nall, dev_vel, gamma);
	kernComputeDotProduct << <dimBlocks, dimThreads >> > (nall, dev_vel, ke_idata);
	ke = reduce_energy_wrapper(ke_idata, ke_odata, nall);

	ke *= 0.5 * m;
	T = ke / (1.5 * float(nall) * kB);

	cudaMemset(dev_force, 0, sizeof(glm::vec3) * nall);
	cudaMemset(dev_pe, 0, sizeof(float));

	float coef = sqrt(24. * tau * m * kB * T0 / dt);
	kernNaiveForce << <dimBlocks, dimThreads >> >(nall, dev_pos, dev_force, dev_vel, coef, hL, L, dcut2, A, B, C, D, m, tau, dev_pe, states);
	cudaMemcpy(&pe, dev_pe, sizeof(float), cudaMemcpyDeviceToHost);
	//////////////////////////////


	//init velocity
	//glm::vec3 mon(0.0f, 0.0f, 0.0f);
	//for (int i = 0; i < nall; i++) {

	//	float a = (float)unitDistrib(rng);
	//	float b = (float)unitDistrib(rng);
	//	float c = (float)unitDistrib(rng);
	//	h_vel[i] = glm::vec3(a,b,c);
	//	mon += h_vel[i];
	//}
	//mon /= nall; ke = 0.0f;
	//for (int i = 0; i < nall; i++) {
	//	h_vel[i] -= mon;
	//	ke += glm::dot(h_vel[i], h_vel[i]);
	//}
	//ke *= 0.5 * m;
	//T = ke / (1.5 * float(nall) * kB);
	//float gama = sqrt(T0 / T);

	//ke = 0.0f;
	//for (int i = 0; i < nall; i++) {
	//	h_vel[i] *= gama;
	//	ke += glm::dot(h_vel[i], h_vel[i]);
	//}

	//ke *= 0.5 * m;
	//T = ke / (1.5 * float(nall) * kB);

	//forceCPU(0);

	printf("ke = %f pe = %f T = %f\n", ke, pe, T);
}

void MD::MD_free()
{
	cudaFree(dev_force);
	cudaFree(dev_pe);
	cudaFree(dev_vel);
	cudaFree(dev_pos);
	cudaFree(states);
	cudaFree(ke_idata);
	cudaFree(ke_odata);

	delete[] h_force;
	delete[] h_pos;
	delete[] h_vel;

	delete M;
	delete rnd;
}

void MD::MD_run()
{
	for (int i = 1; i < N; i++) {
		MD_Loop(i);
	}
}

__global__ void testinitrandom(int n, int seed, curandState *states) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(seed, idx, 0, &states[idx]);
}

__global__ void kernTestRandom(int n, curandState *states, glm::vec3 *vec) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < n) {
		float a = curand_uniform(&states[idx]) - 0.5;
		float b = curand_uniform(&states[idx]) - 0.5;
		float c = curand_uniform(&states[idx]) - 0.5;
		vec[idx] = glm::vec3(a, b, c);
	}

}

void MD::MD_Loop(int k)
{
	int dimThreads = threads;
	int dimBlocks = (nall + dimThreads - 1) / (dimThreads);

	for (int i = 0; i < nstep; i++) {
		kernNaiveVelocityIntegration << <dimBlocks, dimThreads >> > (nall, dev_vel, dev_force, inv_m, hdt);

		kernNaivePositionIntegration << <dimBlocks, dimThreads >> > (nall, dev_pos, dev_vel, dt);

		cudaMemset(dev_force, 0.0f, sizeof(glm::vec3) * nall);
		cudaMemset(dev_pe, 0.0f, sizeof(float));
		double TT = T0 + (Tt - T0) / (N - 1) * k;
		float coef = sqrt(24. * tau * m * kB * TT / dt);
		kernNaiveForce << <dimBlocks, dimThreads >> >(nall, dev_pos, dev_force, dev_vel, coef, hL, L, dcut2, A, B, C, D, m, tau, dev_pe, states);

		kernNaiveVelocityIntegration << <dimBlocks, dimThreads >> > (nall, dev_vel, dev_force, inv_m, hdt);

		if (i % ifreq == 0) {
			kernComputeDotProduct << <dimBlocks, dimThreads >> > (nall, dev_vel, ke_idata);
			ke = reduce_energy_wrapper(ke_idata, ke_odata, nall);

			ke *= 0.5 * m; t += dt;
			T = ke / (1.5 * float(nall) * kB);

			cudaMemcpy(&pe, dev_pe, sizeof(float), cudaMemcpyDeviceToHost);
			printf("step %d ke %f pe %f T %f TT %f coef %f\n", i, ke, pe, T, TT, coef);
		}

	}
	//for (int i = 0; i < nstep; i++) {

	//	for (int j = 0; j < nall; j++) {
	//		h_vel[j] += h_force[j] * hdt;
	//	}

	//	for (int j = 0; j < nall; j++) {
	//		h_pos[j] += h_vel[j] * hdt;
	//	}

	//	float TT = T0;
	//	float Gc = (Tt - T0) / (N - 1);
	//	TT = TT + k * Gc;
	//	pe = 0.0;
	//	float coef = sqrt(24. * tau * m * kB * TT / dt);//coef to calculate w
	//	forceCPU(k);

	//	for (int j = 0; j < nall; j++) {
	//		h_vel[j] += h_force[j] * hdt;
	//	}

	//	ke = 0.0f;
	//	for (int i = 0; i < nall; i++) {
	//		ke += glm::dot(h_vel[i], h_vel[i]);
	//	}

	//	ke *= 0.5 * m;
	//	T = ke / (1.5 * float(nall) * kB);

	//	if (i % ifreq == 0) printf("step %d ke %f pe %f T %f TT %f coef %f\n", i, ke, pe, T, TT, coef);
	//}
}
