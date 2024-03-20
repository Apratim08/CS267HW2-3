#include "common.h"
#include <cuda.h>
#include<thrust/scan.h>
#include<thrust/execution_policy.h>
#include<cmath>
#include <bits/stdc++.h>
#include <iostream>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int bin_blks; // blocks for bin operations

int num_bins;
double bin_size;
int* sorted_parts; // separate array recording id of particles sorted by bin index
int* bin_start_idx; // start idx of the bins in sorted_parts
int* dynamic_assign_idx; // a temporal array for updating sorted_parts
int* bin_count;

// Initialize variables in the GPU
static void gpu_init_arrays(int num_parts, double size){
    num_bins = (int)floor(size / cutoff);
    int gpu_nof_binTot = num_bins * num_bins;
    cudaMalloc((void**)& sorted_parts, num_parts * sizeof(int));
    cudaMalloc((void**)& bin_start_idx, (gpu_nof_binTot + 1) * sizeof(int));
    cudaMalloc((void**)& bin_count, gpu_nof_binTot * sizeof(int));
    cudaMalloc((void**)& dynamic_assign_idx, (gpu_nof_binTot + 1) * sizeof(int));
}

// set initialize values for empty arrays
__global__ void set_int_array(int* array, int val, int array_len){
    // Get the thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= array_len)
        return;
    array[tid] = val;
}

// update bin_count
__global__ void update_bin_count(particle_t* particles, int* bin_start_idx, double bin_size, int num_bins, int num_parts) {
    // printf(":)");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts){
        return;
    }
    // printf("Thread %d\n", tid);
    int bx = (int)floor(particles[tid].x / bin_size);
    int by = (int)floor(particles[tid].y / bin_size);
    int bin_index = bx + by * num_bins;
    atomicAdd(&bin_start_idx[bin_index], 1);
}

// put order of particles to sorted_parts
__global__ void update_sorted_parts(particle_t* particles, int* sorted_parts, int* dynamic_assign_idx, double bin_size, int num_bins, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int bx = (int)floor(particles[tid].x / bin_size);
    int by = (int)floor(particles[tid].y / bin_size);
    int bin_index = bx + by * num_bins;
    int write_index = atomicAdd(&dynamic_assign_idx[bin_index], 1);
    // printf("write_index %d\n", write_index);
    // if (write_index >= num_parts) {
    //     printf("Thread %d calculated out-of-bounds bin_index: %d\n", tid, bin_index);
    //     return;
    // }
    sorted_parts[write_index] = tid;
}

__global__ void setZero(int* arr, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) {
        return;
    }
    arr[tid] = 0;
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}


__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
}

// O(N) Code compute_forces_gpu() function
__global__ void compute_forces_gpu_ON(particle_t* particles, int num_parts, int* sorted_parts, int* bin_start_idx, double bin_size, int num_bins) {
    // Get the thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = 0;
    particles[tid].ay = 0;
    int bx = particles[tid].x / bin_size;
    int by = particles[tid].y / bin_size;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int nbx = bx + dx;
            int nby = by + dy;

            // Check if the neighboring bin is valid
            if (nbx >= 0 && nbx < num_bins && nby >= 0 && nby < num_bins) {
                int neighbor_bin_index = nbx + nby * num_bins;
                // find start_idx and end_idx of the neighbor bin in sorted_parts
                int cur_bin_start_idx = bin_start_idx[neighbor_bin_index];
                int cur_bin_end_idx = bin_start_idx[neighbor_bin_index + 1];
                for (int i =  cur_bin_start_idx; i < cur_bin_end_idx; ++i) {
                    apply_force_gpu(particles[tid], particles[sorted_parts[i]]);
                }
            }
        }
    }
}

// Starter Code move_gpu() function
__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    num_bins = (int)floor(size / cutoff);
    bin_size = size / num_bins;
    // init arrays
    gpu_init_arrays(num_parts, size);
    bin_blks = (num_bins * num_bins + NUM_THREADS - 1) / NUM_THREADS;
    int bin_blks_start = (num_bins * num_bins + 1 + NUM_THREADS - 1) / NUM_THREADS;
    // set bin_start_idx with default values num_parts
    set_int_array<<<bin_blks_start, NUM_THREADS>>>(bin_start_idx, num_parts, num_bins * num_bins + 1);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    // set bin_count with default values 0
    set_int_array<<<bin_blks, NUM_THREADS>>>(bin_count, 0, num_bins * num_bins);
    // update bin_count
    update_bin_count<<<blks, NUM_THREADS>>>(parts, bin_count, bin_size, num_bins, num_parts);
    // do prefix sum and store to bin_start_idx, note bin_start_idx has length num_bins * num_bins + 1,
    // this update 0:-1 entries, where -1 entry left as default value "num_parts"
    thrust::exclusive_scan(thrust::device, bin_count, bin_count + num_bins * num_bins, bin_start_idx);
    // copy bin_start_idx to dynamic_assign_idx
    cudaMemcpy(dynamic_assign_idx, bin_start_idx, num_bins * num_bins * sizeof(int), cudaMemcpyDeviceToDevice);
    // update sorted_parts using dynamic_assign_idx, dynamic_assign_idx is updated after this, bin_start_idx
    // does not change
    update_sorted_parts<<<blks, NUM_THREADS>>>(parts, sorted_parts, dynamic_assign_idx, bin_size, num_bins, num_parts);
    // Compute forces
    compute_forces_gpu_ON<<<blks, NUM_THREADS>>>(parts, num_parts, sorted_parts, bin_start_idx, bin_size, num_bins);
    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}