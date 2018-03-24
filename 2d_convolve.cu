#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cfloat>
#include <utility>

// #define USE_EXAMPLE /* uncomment to use fixed example instead of generating vectors */
// #define PRINT_ANSWERS /* uncomment to print out solution */

const int threads_per_block = 256;


// Forward function declarations
/*
Task: Perform a 2D convolution on an NxN matrix A_in laid out in memory
with contiguous rows and a KxK filter F. K must be odd. Put
the result in A_out.

With this layout, A row i column j is A[i * N + j]; and
F row i column j is F[i * N + j].

When elements outside A are required apply this by "extending"
the edge (see also description in the Wikipedia article), substituting
the closest value within A. For example, if accessing row -1, col 3 of A
is required, instead substitute row 0, col 3 of A (if As rows numbered
from 0).

This means that for any given element in A, select a KxK neighborhood
around it, and multiply it element-by-element with the KxK filter,
then sum the results and put the sum in the corresponding element of A_out.

For example, if K = 3, then this means that 


  // A_out(row i, col j) = 
  //   F(row 0, col 0) * A_in(row MAX(i - 1, 0), col j - 1) + ...
  A_out[i * N + j] = 
     F[0 * K + 0] * A_in[MAX(i - 1, 0) * N + MAX(j - 1, 0)] + 
     F[0 * K + 1] * A_in[MAX(i - 1, 0) * N + j] + 
     F[0 * K + 2] * A_in[MAX(i - 1, 0) * N + MIN(j + 1, N-1)] + 

     F[1 * K + 0] * A_in[i * N + MAX(j - 1, 0)] + 
     F[1 * K + 1] * A_in[i * N + j] + 
     F[1 * K + 2] * A_in[i * N + MIN(j + 1, N)] + 
     
     F[2 * K + 0] * A_in[MIN(i + 1, N-1) * N + MAX(j - 1, 0)] + 
     F[2 * K + 1] * A_in[MIN(i + 1, N-1) * N + j] + 
     F[2 * K + 2] * A_in[MIN(i + 1, N-1) * N + MIN(j + 1, N-1)];

See also:
    - CPU_convolve() which implements this below.
    - https://en.wikipedia.org/wiki/Kernel_(image_processing) 
    - https://docs.gimp.org/en/plug-in-convmatrix.html

 */
void GPU_convolve(float *A_in, float *A_out, int N, float *F, int K, int kernel_code, float *kernel_time, float *transfer_time);
void CPU_convolve(float *A_in, float *A_out, int N, float *F, int K);
float *get_random_vector(int N);
float *get_increasing_vector(int N);
float usToSec(long long time);
long long start_timer();
long long stop_timer(long long start_time, const char *name);
void printMatrix(float *X, int N, int M); // utility function you can use
void die(const char *message);
void checkError();

// Main program
int main(int argc, char **argv) {

    //default kernel
    int kernel_code = 1;
    
    // Parse vector length and kernel options
    int N, K;
#ifdef USE_EXAMPLE
    if(argc == 1) {
    } else if (argc == 3 && !strcmp(argv[1], "-k")) {
        kernel_code = atoi(argv[2]); 
        printf("KERNEL_CODE %d\n", kernel_code);
    } else {
        die("USAGE: ./2d_convolve -k <kernel_code> # uses hardcoded example");
    }
#else
    if(argc == 3) {
        N = atoi(argv[1]); // user-specified value
        K = atoi(argv[2]); // user-specified value
    } else if (argc == 5 && !strcmp(argv[3], "-k")) {
        N = atoi(argv[1]); // user-specified value
        K = atoi(argv[2]); // user-specified value
        kernel_code = atoi(argv[4]); 
        printf("KERNEL_CODE %d\n", kernel_code);
    } else {
        die("USAGE: ./2d_convolve <N> <K> -k <kernel_code> # image is NxN, filter is KxK");
    }
#endif

    // Seed the random generator (use a constant here for repeatable results)
    srand(10);

    // Generate random matrices
    long long vector_start_time = start_timer();
#ifdef USE_EXAMPLE
    float A_in[25] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    N = 5;
    K = 3;
    float F[9] = {
        1, 1, 0,
        0, 0, 0,
        0, 0, -1,
    };
#else
    float *A_in = get_random_vector(N * N);
    float *F = get_random_vector(K * K);
    for (int i = 0; i < K * K; ++i) {
        F[i] /= K * K;
    }
#endif
    stop_timer(vector_start_time, "Vector generation");
    float *A_out_GPU;
    float *A_out_CPU;
    cudaMallocHost((void **) &A_out_GPU, N * N * sizeof(float));
    cudaMallocHost((void **) &A_out_CPU, N * N * sizeof(float));
    memset(A_out_CPU, 0, N * N * sizeof(float));
    memset(A_out_GPU, 0, N * N * sizeof(float));


    int num_blocks = (int) ((float) (N + threads_per_block - 1) / (float) threads_per_block);
    int max_blocks_per_dimension = 65535;
    int num_blocks_y = (int) ((float) (num_blocks + max_blocks_per_dimension - 1) / (float) max_blocks_per_dimension);
    int num_blocks_x = (int) ((float) (num_blocks + num_blocks_y - 1) / (float) num_blocks_y);
    dim3 grid_size(num_blocks_x, num_blocks_y, 1);

	
    // Compute the max on the GPU
    float GPU_kernel_time = INFINITY;
    float transfer_time = INFINITY;
    long long GPU_start_time = start_timer();
    GPU_convolve(A_in, A_out_GPU, N, F, K, kernel_code, &GPU_kernel_time, &transfer_time);
    long long GPU_time = stop_timer(GPU_start_time, "\n            Total");
	
    printf("%f\n", GPU_kernel_time);
    
    // Compute the max on the CPU
    long long CPU_start_time = start_timer();
    CPU_convolve(A_in, A_out_CPU, N, F, K);
    long long CPU_time = stop_timer(CPU_start_time, "\nCPU");
  
  
    
 
#ifndef USE_EXAMPLE 
    // Free matrices 
    cudaFree(A_in);
    cudaFree(F);
#endif

    // Compute the speedup or slowdown
    //// Not including data transfer
    if (GPU_kernel_time > usToSec(CPU_time)) printf("\nCPU outperformed GPU kernel by %.2fx\n", (float) (GPU_kernel_time) / usToSec(CPU_time));
    else                     printf("\nGPU kernel outperformed CPU by %.2fx\n", (float) usToSec(CPU_time) / (float) GPU_kernel_time);

    //// Including data transfer
    if (GPU_time > CPU_time) printf("\nCPU outperformed GPU total runtime (including data transfer) by %.2fx\n", (float) GPU_time / (float) CPU_time);
    else                     printf("\nGPU total runtime (including data transfer) outperformed CPU by %.2fx\n", (float) CPU_time / (float) GPU_time);

#ifdef PRINT_ANSWERS
    printf("CPU result:\n");
    printMatrix(A_out_CPU, N, N);
    printf("GPU result:\n");
    printMatrix(A_out_GPU, N, N);
#endif

    // Check the correctness of the GPU results
    float max_delta = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float cpu = A_out_CPU[i];
        float gpu = A_out_GPU[i];
        float delta = fabs(gpu - cpu);
        if (delta > max_delta) {
            /* printf("%f/%f/%f\n", gpu, cpu, real); */
            max_delta = delta;
        }
    }
    cudaFree(A_out_CPU);
    cudaFree(A_out_GPU);
    /* This should be lenient enough to allow additions/substractions to occur in a different order */
    int wrong = max_delta > 1e-6 * 2 * K;
    // Report the correctness results
    if(wrong) printf("GPU output did not match CPU output (max error %.2f%%)\n", max_delta * 100.);
}


static int clamp(int x, int low, int high) {
    if (x < low)
        return low;
    else if (x > high)
        return high;
    else
        return x;
}

//   paralell 
/*__global__ void vector_max_kernelX(float *A_in, float *A_out, int N1, float *F, int K1, int offset1) {

    // Determine the "flattened" block id and thread id
    int block_id = blockIdx.x + gridDim.x * blockIdx.y;
    int thread_id = blockDim.x * block_id + threadIdx.x;
    int thread_i;
    int i = thread_id;
    __shared__ int N, K, offset;
    K=K1;
    N=N1;
    offset = offset1;
    int a,b;
    __shared__ int min_offset, max_offset;
    min_offset = -(K-1) / 2;
    max_offset = (K-1) / 2;
    int ii = thread_i/K - (K * K / 2);
    int jj = thread_i%K - (K * K / 2);
    float *F_center = &F[ K * K / 2 ];
    float result = 0.0;
    if (i+ii<0) {
        if (j+jj<0)
             result += A_in[0] * F_center[ii * K +jj];
        else if (j+jj>N-1)
             result += A_in[N-1] * F_center[ii * K +jj];
        else
             result += A_in[j+jj] * F_center[ii * K +jj];
    }
    else if (i+ii> N-1) {
        if (j+jj<0)
             result += A_in[(N-1)*N] * F_center[ii * K +jj];
        else if(j+jj>N-1)
             result += A_in[(N-1)*N+N-1] * F_center[ii * K +jj];
        else
             result += A_in[(N-1)*N+j+jj] * F_center[ii * K +jj];
    }
    else {
        if (j+jj<0)
             result += A_in[i+ii] * F_center[ii * K +jj];
        else if(j+jj>N-1)
             result += A_in[(i+ii)*N+N-1] * F_center[ii * K +jj];
        else
             result += A_in[(i+ii)*N+j+jj] * F_center[ii * K +jj];
    }
    A_out[i * N + j ] += result;
    
}*/



__global__ void vector_max_kernel(float *A_in, float *A_out, int N1, float *F, int K1) {

    // Determine the "flattened" block id and thread id
     int block_id = blockIdx.x + gridDim.x * blockIdx.y;
     int thread_id = blockDim.x * block_id + threadIdx.x; 
     __shared__ int N, K;
/*     __shared__ float   F_s[256],A_in_s[256];
     A_in_s[thread_id_lo] = A_in[thread_id];
     F_s[thread_id_lo] = F[thread_id];
     __syncthreads();*/
    
     K=K1;
     N=N1;
   
      int a,b;
     __shared__ int min_offset, max_offset;
     min_offset = -(K-1) / 2;
     max_offset = (K-1) / 2;

     float *F_center = &F[ K * K / 2 ];
      
     int  i,j;
     i=thread_id/N;
     j=thread_id%N;

     float result = 0.0;
     for (int ii = min_offset; ii <= max_offset; ++ii) {
             for (int jj = min_offset; jj <= max_offset; ++jj) {

                        
                        if (i+ii < 0)
                             a=0;
                        else if (i+ii > N-1)
                             a=N-1;
                        else
                             a=i+ii;
                       
                        if (j+jj < 0)
                             b=0;
                        else if (j+jj > N-1)
                             b=N-1;
                        else
                             b=j+jj;
                    
                                result += A_in[a*N + b] *
                              F_center[ii * K +jj];
              }
      }
                A_out[i * N + j ] = result;
   
}



void GPU_convolve(float *A_in, float *A_out, int N, float *F, int K, int kernel_code, float *kernel_runtime, float *transfer_runtime) {
    // IMPLEMENT YOUR BFS AND TIMING CODE HERE
    long long transfer_time = 0;
    long long kernel_time = 0;

    int A_size = N * N * sizeof(float);
    int F_size = K * K * sizeof(float);

/*    // Allocate CPU memory for the result
    float *out_CPU;
    cudaMallocHost((void **) &out_CPU, A_size * sizeof(float));
    if (out_CPU == NULL) die("Error allocating CPU memory");
*/
    // Allocate GPU memory for the inputs and the result
    long long memory_start_time = start_timer();

    float *A_GPU, *out_GPU, *F_GPU;
    if (cudaMalloc((void **) &A_GPU, A_size) != cudaSuccess) die("Error allocating GPU memory");
    if (cudaMalloc((void **) &out_GPU, A_size) != cudaSuccess) die("Error allocating GPU memory");
    if (cudaMalloc((void **) &F_GPU, F_size) != cudaSuccess) die("Error allocating GPU memory");	

    // Transfer the input vectors to GPU memory
    cudaMemcpy(A_GPU, A_in, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(F_GPU, F, F_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();  // this is only needed for timing purposes
    transfer_time += stop_timer(memory_start_time, "\nGPU:\t  Transfer to GPU");
	
    // Determine the number of thread blocks in the x- and y-dimension
    int num_blocks = (int) ((float) (N*N + threads_per_block - 1) / (float) threads_per_block);
    int max_blocks_per_dimension = 65535;
    int num_blocks_y = (int) ((float) (num_blocks + max_blocks_per_dimension - 1) / (float) max_blocks_per_dimension);
    int num_blocks_x = (int) ((float) (num_blocks + num_blocks_y - 1) / (float) num_blocks_y);
    dim3 grid_size(num_blocks_x, num_blocks_y, 1);
	
    // Execute the kernel to compute the vector sum on the GPU
    long long kernel_start_time;
    kernel_start_time = start_timer();
   

    vector_max_kernel <<< grid_size , threads_per_block >>> (A_GPU, out_GPU,  N, F_GPU, K);


    cudaDeviceSynchronize();  // this is only needed for timing purposes
    kernel_time += stop_timer(kernel_start_time, "\t Kernel execution");
    
    checkError();
    
    // Transfer the result from the GPU to the CPU
    memory_start_time = start_timer();
    
    //copy C back
    cudaMemcpy(A_out, out_GPU, A_size, cudaMemcpyDeviceToHost);
    checkError();
    cudaDeviceSynchronize();  // this is only needed for timing purposes
    transfer_time += stop_timer(memory_start_time, "\tTransfer from GPU");
    			    
    // Free the GPU memory
    cudaFree(A_GPU);
    cudaFree(out_GPU);
    cudaFree(F_GPU);

    // fill input pointers with ms runtimes
    *kernel_runtime = usToSec(kernel_time);
    *transfer_runtime = usToSec(transfer_time);
    //return a single statistic
   // return 0;


}




void CPU_convolve(float *A_in, float *A_out, int N, float *F, int K) {
    int min_offset = -(K-1) / 2;
    int max_offset = (K-1) / 2;

    float *F_center = &F[ K * K / 2 ];
    // If K = 5, F_center points (row 2, col 2), so F_center[1] is (row 2, col 3);
    // F_center[-K] is (row 1, col 2) F_center[K + 1] is (row 3, col 3), etc.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float result = 0.0;
            for (int ii = min_offset; ii <= max_offset; ++ii) {
                for (int jj = min_offset; jj <= max_offset; ++jj) {
                    result += A_in[clamp(i+ii, 0, N-1)*N + clamp(j+jj, 0, N-1)] *
                              F_center[ii * K + jj];
                }
            }
            A_out[i * N + j ] = result;
        }
    }
}


// Returns a randomized vector containing N elements
// This verison generates vector containing values in the range [0,2)
float *get_random_vector(int N) {
    if (N < 1) die("Number of elements must be greater than zero");
	
    // Allocate memory for the vector
    float *V;
    cudaMallocHost((void **) &V, N * sizeof(float));
    if (V == NULL) die("Error allocating CPU memory");
	
    // Populate the vector with random numbers
    for (int i = 0; i < N; i++) V[i] = rand() * 2.0f / RAND_MAX;
	
    // Return the randomized vector
    return V;
}

void printMatrix(float *X, int N, int M) {
    for (int i = 0; i < N; ++i) {
        printf("row %d: ", i);
        for (int j = 0; j < M; ++j) {
            printf("%f ", X[i * M + j]);
        }
        printf("\n");
    }
}

void checkError() {
    // Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (error) {
        char message[256];
        sprintf(message, "CUDA error: %s", cudaGetErrorString(error));
        die(message);
    }
}

// Returns the current time in microseconds
long long start_timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// converts a long long ns value to float seconds
float usToSec(long long time) {
    return ((float)time)/(1000000);
}

// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, const char *name) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
    float elapsed = usToSec(end_time - start_time);
    printf("%s: %.5f sec\n", name, elapsed);
    return end_time - start_time;
}


// Prints the specified message and quits
void die(const char *message) {
    printf("%s\n", message);
    exit(1);
}
