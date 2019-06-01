#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>

/* Store all parameters passed as argument to main */
struct aux_args {
    float c0_real; /* Real part of c0 */
    float c0_imag; /* Imaginary part of c0 */
    float c1_real; /* Real part of c1 */
    float c1_imag; /* Imaginary part of c1 */
    unsigned w; /* width, in pixels, of the image to be generated */
    unsigned h; /* high, in pixels, of the image to be generated */
    unsigned is_cpu; /* True if the programs needs to run in the cpu */
    unsigned threads; /* Number of threads to be executed */
    unsigned blocks; /* Number of threads to be executed */
};

/* MACROS */
#define MAX 100 /* Max number of iterations */

/* Error handle */
#define ERRO(...) { \
        fprintf(stderr, __VA_ARGS__); \
        exit(EXIT_FAILURE); \
}

/* CUDA Error handle */
#define cudaERRO(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            ERRO(msg); \
        } \
    } while (0)

/* Parse data from argv[] to struct */
static struct aux_args* parse_args(const char* argv[]) {

    /* Static keyword will keep data until end of exec */
    static struct aux_args ret;

    ret.c0_real = strtof (argv[1], NULL);
    ret.c0_imag = strtof (argv[2], NULL);
    ret.c1_real = strtof (argv[3], NULL);
    ret.c1_imag = strtof (argv[4], NULL);
    ret.w = strtol (argv[5], NULL, 10);
    ret.h = strtol (argv[6], NULL, 10);
    ret.is_cpu = (strcmp(argv[7], "cpu") == 0);
    ret.threads = strtol (argv[8], NULL, 10);
    ret.blocks = strtol (argv[9], NULL, 10);

    return &ret;
}

/* --- CPU --- */

/* Auxiliar function for complex module */
inline float cpu_squared_moduleZ (float z_real, float z_imag) {
    return z_real*z_real + z_imag*z_imag;
}

/* Iterate using maldelbrot rule in each potion */
unsigned cpu_iterate (float c_real, float c_imag) {
	float z_real = 0.0;
	float z_imag = 0.0;
	unsigned iter = 0;

	while (cpu_squared_moduleZ(z_real, z_imag) <= 4.0 && iter < MAX ) {

        /* Calculating new position */
        float nz_r = z_real*z_real - z_imag*z_imag + c_real;
		float nz_i = 2.0*z_real*z_imag + c_imag;
        z_real = nz_r;
        z_imag = nz_i;
		++iter;
	}
	return iter;
}

/* Receive input arguments in arg, an array calc, of size WxH, put the number
   of iterations corresponding to each pixel of image */
void cpu_make_iterations (struct aux_args *arg, int *calc) {

    unsigned long long int size = arg->w * arg->h;
    /* Variation of each iteration */
	float dx = (arg->c1_real - arg->c0_real)/(float)arg->w;
	float dy = (arg->c1_imag - arg->c0_imag)/(float)arg->h;

    #pragma omp parallel for num_threads(arg->threads)
	for (unsigned long long int i = 0; i < size; ++i) {

        /* Calculating the complex value of each postion */
		float x = ((i % arg->w) * dx) + arg->c0_real;
		float y = ((i / arg->w) * dy) + arg->c0_imag;
		calc[i] = cpu_iterate (x, y);
	}
}


/* --- GPU --- */

/* Iterate using maldelbrot rule in each potion */
__device__
unsigned gpu_iterate (float c_real, float c_imag) {
    float z_real = 0.0;
    float z_imag = 0.0;
    unsigned iter = 0;

    while ( (z_real*z_real + z_imag*z_imag) <= 4.0 && iter < MAX ) {

        /* Calculating new position */
        float nz_r = z_real*z_real - z_imag*z_imag + c_real;
        float nz_i = 2.0*z_real*z_imag + c_imag;
        z_real = nz_r;
        z_imag = nz_i;
        ++iter;
    }
    return iter;
}

/* Kernel for the execution */
__global__
void gpu_make_iterations (struct aux_args *arg, int *calc) {

    unsigned long long int size = arg->w * arg->h;
    /* Variation of each iteration */
	float dx = (arg->c1_real - arg->c0_real)/(float)arg->w;
	float dy = (arg->c1_imag - arg->c0_imag)/(float)arg->h;

    /* Index of this thread */
    unsigned long long int t_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    /* Position index this thread is taking care of */
    unsigned long long int p_index = t_index;

    while (p_index < size) {

        /* Calculating the complex value of each postion */
    	float x = ((p_index % arg->w) * dx) + arg->c0_real;
    	float y = ((p_index / arg->w) * dy) + arg->c0_imag;
    	calc[p_index] = gpu_iterate(x, y);

        /* Moving to next session */
        p_index += (blockDim.x*gridDim.x);

    }

    return;
}

int main(int argc, const char* argv[]) {

    /* GOOD POSITIONS TO USE */
    /* ./mbrot -2.0 1.5 1.0 -1.5 1000 1000 <...> */

    /* Used for testing. If true, program will print the elapsed time. */
    struct timeval start, end;

    /* Getting parameters */
    struct aux_args *arg = parse_args(argv);

    /* Used for iteration calculus */
	int *calc = (int *)malloc(arg->w * arg->h * sizeof (int));

    if (calc == NULL)
        ERRO("Could not allocate malloc in main().\n");

    /* Time in the start of processing */
    gettimeofday(&start, NULL);

    if (arg->is_cpu)
	    cpu_make_iterations (arg, calc);

    else {
        struct aux_args *d_arg;
        int *d_calc;

        cudaSetDevice(0);

        cudaMalloc((void **)&d_arg, sizeof(struct aux_args));
        cudaMalloc((void **)&d_calc, sizeof(int) * arg->w * arg->h);
        cudaERRO("cudaMalloc failure");

        cudaMemcpy(d_arg, arg, sizeof(struct aux_args), cudaMemcpyHostToDevice);
        cudaERRO("cudaMemcpy H2D failure\n");

        /* <<< blocks , threads >>> */
        gpu_make_iterations<<<arg->blocks, arg->threads>>>(d_arg, d_calc);
        cudaERRO("Kernel launch failure\n");

        /* Wait for all blocks/threads finishes their work */
        cudaDeviceSynchronize();
        cudaERRO("cudaDeviceSynchronize failure\n");

        cudaMemcpy(calc, d_calc, arg->w*arg->h*sizeof(int), cudaMemcpyDeviceToHost);
        cudaERRO("cudaMemcpy D2H failure\n");

        cudaFree(d_calc);
        cudaFree(d_arg);
    }

    /* Final time */
    gettimeofday(&end, NULL);

    /* Getting time */
    double elapsed_time = (end.tv_sec - start.tv_sec) +
            (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("%.4f\n", elapsed_time);

    /* Let it go, let it go */
    free(calc);

    return 0;
}
