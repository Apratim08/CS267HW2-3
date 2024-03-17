#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <iostream>
#include <cstdlib>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int num_bins;
double bin_size;
int* sorted_parts; // separate array recording id of particles sorted by bin index
int* bin_start_idx; // start idx of the bins in sorted_parts
int* dynamic_assign_idx; // a temporal array for updating sorted_parts

// get bin_start_idx
__global__ void update_bin_start_idx(particle_t* particles, int* bin_start_idx, double bin_size, int num_bins, int num_parts) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int bx = (int)floor(particles[tid].x / bin_size);
    int by = (int)floor(particles[tid].y / bin_size);
    int bin_index = bx + by * num_bins;
    if (bin_index < 0 || bin_index >= num_bins * num_bins) {
        printf("Thread %d calculated out-of-bounds bin_index: %d\n", tid, bin_index);
        return;
    }
    atomicAdd(bin_start_idx + bin_index, 1);
}

// assign particles to bins based on their corrent position, put their id / ori index in sorted_parts
__global__ void update_sorted_parts(particle_t* particles, int* sorted_parts, int* dynamic_assign_idx, double bin_size, int num_bins, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int bx = particles[tid].x / bin_size;
    int by = particles[tid].y / bin_size;
    int bin_index = bx + by * num_bins;
    
    int write_index = atomicAdd(&dynamic_assign_idx[bin_index], 1);
    sorted_parts[write_index] = tid;
}

static void gpu_init_arrays(int num_parts, int num_bins) {
    // Initialize sorted_parts, bin_start_idx and dynamic_assign_idx
    cudaMalloc((void**)& sorted_parts, num_parts * sizeof(int));
    cudaMalloc((void**)& bin_start_idx, num_bins * num_bins * sizeof(int));
    cudaMalloc((void**)& dynamic_assign_idx, num_bins * num_bins * sizeof(int));
}

static void gpu_clear_arrays() {
    // Free memory allocated for sorted_parts, bin_start_idx and dynamic_assign_idx
    cudaFree(sorted_parts);
    cudaFree(bin_start_idx);
    cudaFree(dynamic_assign_idx);
    
    // cudaMalloc(bin_start_idx);
    // cudaMalloc(dynamic_assign_idx);
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

    
    for (int j = 0; j < num_parts; j++) {
        apply_force_gpu(particles[tid], particles[j]);
    }
}

// O(N) Code compute_forces_gpu() function
__global__ void compute_forces_gpu_ON(particle_t* particles, int num_parts, int* sorted_parts, int* bin_start_idx, double bin_size, int num_bins) {
    // Get the thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cur_bin_end_idx; 
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
                if (neighbor_bin_index + 1 < num_bins * num_bins) {
                    cur_bin_end_idx = bin_start_idx[neighbor_bin_index + 1];
                } else {
                    cur_bin_end_idx = num_parts;
                }

                // only compute force if the neighbor bin is not empty
                if (cur_bin_start_idx < cur_bin_end_idx) {
                    // Iterate over particles in the neighboring bin
                    for (int i =  cur_bin_start_idx; i < cur_bin_end_idx; ++i) {
                        apply_force_gpu(particles[tid], particles[sorted_parts[i]]);
                    }
                }
            }
        }
    }
}

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

    // NOTE: This means that blks * NUM_THREADS >= num_parts, e.g. each particle
    // would have 1 thread to do computation.
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // Initialize bin size and count (note on x and y axis 
    // these two numbers should be the same)
    num_bins = (int)floor(size / cutoff);
    bin_size = size / num_bins;

    std::cout << "num_bins " << num_bins << std::endl;
    std::cout << "bin_size " << bin_size << std::endl;

    gpu_init_arrays(num_parts, num_bins);

    std::cout << "Arrays initialized successfully." << std::endl;

    for (int i = 0; i < num_parts; ++i) {
        std::cout << "Particle " << i << std::endl;
        int bx = (int)floor(parts[i].x / bin_size);
        int by = (int)floor(parts[i].y / bin_size);
        int bin_index = bx + by * num_bins;
        std::cout << "Particle " << i << "has bx " << bx << "by " << by << "bin_index" << bin_index << std::endl;
    }
    exit();

    // This should update bin_start_idx to contain number of particles at each bin
    update_bin_start_idx<<<blks, NUM_THREADS>>>(parts, bin_start_idx, bin_size, num_bins, num_parts);
    
    std::cout << "update_bin_start_idx " << bin_start_idx[0] << std::endl;

    // in-place prefix sum
    thrust::exclusive_scan(bin_start_idx, bin_start_idx + num_bins * num_bins, bin_start_idx);
    
    std::cout << "prefix sum " << bin_start_idx[1] << std::endl;

    // copy data from bin_start_idx to dynamic_assign_idx
    cudaMemcpy(dynamic_assign_idx, bin_start_idx, num_bins * num_bins * sizeof(int), cudaMemcpyDeviceToDevice);
    
    std::cout << "copy " << dynamic_assign_idx[1] << std::endl;

    // get sorted_parts based on dynamic_assign_idx
    update_sorted_parts<<<blks, NUM_THREADS>>>(parts, sorted_parts, dynamic_assign_idx, bin_size, num_bins, num_parts);

    std::cout << "sorted_parts " << sorted_parts[dynamic_assign_idx[1]] << std::endl;
    
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // compute the number of blocks needed
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // Compute forces
    compute_forces_gpu_ON<<<blks, NUM_THREADS>>>(parts, num_parts, sorted_parts, bin_start_idx, bin_size, num_bins);
    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    // Clear array from this iter
    gpu_clear_arrays();
    // Get new empty arrays for next iter
    gpu_init_arrays(num_parts, num_bins);
    // This should update bin_start_idx to contain number of particles at each bin
    update_bin_start_idx<<<blks, NUM_THREADS>>>(parts, bin_start_idx, bin_size, num_bins, num_parts);
    // in-place prefix sum
    thrust::exclusive_scan(bin_start_idx, bin_start_idx + num_bins * num_bins, bin_start_idx);
    // copy data from bin_start_idx to dynamic_assign_idx
    cudaMemcpy(dynamic_assign_idx, bin_start_idx, num_bins * num_bins * sizeof(int), cudaMemcpyDeviceToDevice);
    // get sorted_parts based on dynamic_assign_idx
    update_sorted_parts<<<blks, NUM_THREADS>>>(parts, sorted_parts, dynamic_assign_idx, bin_size, num_bins, num_parts);

}
