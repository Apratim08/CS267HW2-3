#include "common.h"
#include <cuda.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int bin_count;
int* bin_particles;
int* bin_particles_gpu;
double bin_size;
double bin_size_gpu;
int bin_count_gpu;
double size_gpu;
int num_parts_gpu;

particle_t* parts_gpu;
particle_t* separate_parts_gpu; // Separate array for particles sorted by bin index


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

   // Initialize bin size and count
    bin_size = size / sqrt(num_parts);
    bin_count = ceil(size / bin_size);

    // Initialize bin particles count
    bin_particles = (int*)malloc(bin_count * bin_count * sizeof(int));
    memset(bin_particles, 0, bin_count * bin_count * sizeof(int));

    // Iterate through particles and count particles per bin
    for (int i = 0; i < num_parts; ++i) {
        int bin_x = (int)(parts[i].x / bin_size);
        int bin_y = (int)(parts[i].y / bin_size);
        bin_particles[bin_x * bin_count + bin_y]++;
    }

    // Prefix sum the bin counts
    for (int i = 1; i < bin_count * bin_count; ++i) {
        bin_particles[i] += bin_particles[i - 1];
    }

    // Allocate memory for separate array of particles sorted by bin index
    int total_particles = bin_particles[bin_count * bin_count - 1];
    cudaMalloc(&separate_parts_gpu, total_particles * sizeof(particle_t));

    // Allocate memory for bin_particles_gpu and copy data to GPU
    cudaMalloc(&bin_particles_gpu, bin_count * bin_count * sizeof(int));
    cudaMemcpy(bin_particles_gpu, bin_particles, bin_count * bin_count * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate memory for other GPU variables and copy data
    cudaMalloc(&parts_gpu, num_parts * sizeof(particle_t));
    cudaMemcpy(parts_gpu, parts, num_parts * sizeof(particle_t), cudaMemcpyHostToDevice);

    // Set other GPU variables
    bin_size_gpu = bin_size;
    bin_count_gpu = bin_count;
    size_gpu = size;
    num_parts_gpu = num_parts;

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // compute the number of blocks needed
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);


    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);

    // synchronize 
    cudaDeviceSynchronize();
}
