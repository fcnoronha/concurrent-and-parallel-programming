#include "dmbrot.h"

/*

    Program to generate images of the mandelbrot set using a network of computers.

    By Felipe Noronha at IME-USP.

*/

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

/* 'inline' makes the code be 'rewritten' in the call location, making it faster */
/* Transforms an interger value into a RGB color format */
inline void to_RGB(png_byte *ptr, int val) {

    /* Nomalize value */
	float color = ((float)val / MAX) * 255;
	ptr[0] = (int)color;
	ptr[1] = (int)color;
	ptr[2] = (int)color;
}

/* Parse data from argv[] to struct */
static struct aux_args* parse_args(char* argv[]) {

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
    ret.saida = argv[9];

    return &ret;
}

/* Create a png image from a int array calc[], what is an 1D
   representation of a 2D array, where each position is a certain
   number of iterations */
void generate_image(struct aux_args *arg, int *calc) {

    FILE *file_ptr = NULL;
    png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
	png_bytep row = NULL;

    /* Opening new file for writing */
    file_ptr = fopen(arg->saida, "wb");

    if (file_ptr == NULL)
        ERRO("Could not open new file for writing.\n");

    /* Initialize new empty structure for png */
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (png_ptr == NULL)
        ERRO("Could not allocate new struct for png.\n");

    /* Initialize information structure for png */
    info_ptr = png_create_info_struct(png_ptr);

    if (info_ptr == NULL)
        ERRO("Could not allocate information structure for png.\n");

    /* From now on any error will be handled here */
    if (setjmp(png_jmpbuf(png_ptr)))
        ERRO("Error during png creation.\n");

    /* Initialize png_ptr with new file */
    png_init_io(png_ptr, file_ptr);

    /* Header of png for 8-bit colour depth */
    png_set_IHDR(png_ptr,
        info_ptr,
        arg->w,
        arg->h,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE,
        PNG_FILTER_TYPE_BASE);

    /* writing down generated information */
    png_write_info(png_ptr, info_ptr);

    /* Used to build image */
    row = (png_byte *)malloc(3 * arg->w * sizeof(png_byte));

    if (row == NULL)
        ERRO("Could not allocate memory for image building.\n");

    /* write image data, one row at a time */
    for (int y = 0; y < arg->h; y++) {
        for (int x = 0; x < arg->w; x++) {
            to_RGB(&row[x*3], calc[y*arg->w + x]);
        }
        png_write_row(png_ptr, row);
    }

    /* End writing */
    png_write_end(png_ptr, NULL);

    /* Cleaning up */
    fclose(file_ptr);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    free(row);

    return;
}

/* --- CPU --- */

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
void cpu_make_iterations (struct aux_args *arg, int *calc, int cnt) {

    /* Variation of each iteration */
	float dx = (arg->c1_real - arg->c0_real)/(float)arg->w;
	float dy = (arg->c1_imag - arg->c0_imag)/(float)arg->h;

    #pragma omp parallel for num_threads(arg->threads)
	for (int i = 0; i < cnt; ++i) {

        /* Position for the calculus */
        int pos = calc[i];

        /* Calculating the complex value of each postion */
		float x = ((pos % arg->w) * dx) + arg->c0_real;
		float y = ((pos / arg->w) * dy) + arg->c0_imag;
		calc[i] = cpu_iterate (x, y);
	}
}

/* Auxiliar function for complex module */
inline float cpu_squared_moduleZ (float z_real, float z_imag) {
	return z_real*z_real + z_imag*z_imag;
}

/* --- GPU --- */

/* Kernel for the execution */
__global__
void gpu_make_iterations (struct aux_args *arg, int *calc, int cnt) {

    /* Variation of each iteration */
	float dx = (arg->c1_real - arg->c0_real)/(float)arg->w;
	float dy = (arg->c1_imag - arg->c0_imag)/(float)arg->h;

    /* Index of this thread */
    unsigned long long int t_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    /* Position index this thread is taking care of */
    unsigned long long int p_index = t_index;

    while (p_index < cnt) {

        /* position */
        int pos = calc[p_index];

        /* Calculating the complex value of each postion */
    	float x = ((pos % arg->w) * dx) + arg->c0_real;
    	float y = ((pos / arg->w) * dy) + arg->c0_imag;
    	calc[p_index] = gpu_iterate(x, y);

        /* Moving to next session */
        p_index += (blockDim.x*gridDim.x);

    }

    return;
}

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

int main(int argc, char** argv) {

    /* GOOD POSITIONS TO USE */
    /* ./dmbrot -2.0 1.5 1.0 -1.5 1000 1000 cpu 8 ./img.png */

    /* Initialize the MPI environment */
    if (MPI_Init(&argc, &argv))
        ERRO("Not possible initialize MPI environment.\n");

    /* Get the number of mpi-processes */
    int world_size;
    if (MPI_Comm_size(MPI_COMM_WORLD, &world_size))
        ERRO("Not possible to get world_size.\n");

    /* Get the rank of this mpi-process */
    int world_rank;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank))
        ERRO("Not possible to get world_rank.\n");

    /* Id of mpi-process with global resposabilities */
    const int root = 0;

    /* Getting parameters */
    struct aux_args *arg = parse_args(argv);

    /* Total number of pixels in final image */
    int n_pixel = arg->w * arg->h;

    /* Used for iteration calculus */
    int *calc;
    if (world_rank == 0) {
        /* Only root will have */
        calc = (int *)malloc(n_pixel * sizeof (int));
        if (calc == NULL)
            ERRO("Could not allocate malloc in main().\n");

        /* Filling calc[] with real index in given position */
        for (int i = 0; i < n_pixel; i++)
        calc[i] = i;
    }

    /* Represents how many pixels each mpi-process will deal with */
    int *cnt = (int *)malloc(sizeof(int) * world_size);
    int process_with_more_pixel = n_pixel % world_size;
    for (int i = 0; i < world_size; i++){
        cnt[i] = n_pixel / world_size;
        cnt[i] += (i < process_with_more_pixel);
    }

    /* How many pixels this mpi-process will deal with */
    int this_cnt = cnt[world_rank];

    /* Local data of each mpi-process */
    int *calc_local = (int *)malloc(this_cnt * sizeof (int));

    /* The starting point of everyone's data in the calc array */
    int *starts = (int *)malloc(sizeof(int) * world_size);
    int aux = 0;
    for (int i = 0; i < world_size; i++) {
        starts[i] = aux;
        aux += cnt[i];
    }

    /* Dividing data through mpi-processes */
    MPI_Scatterv (calc, cnt, starts,
                MPI_INT,
                calc_local, this_cnt, MPI_INT,
                root, MPI_COMM_WORLD);


    if (arg->is_cpu)
	    cpu_make_iterations (arg, calc_local, this_cnt);

    else {
        struct aux_args *d_arg;
        int *d_calc;

        cudaSetDevice(0);

        cudaMalloc((void **)&d_arg, sizeof(struct aux_args));
        cudaMalloc((void **)&d_calc, sizeof(int) * this_cnt);
        cudaERRO("cudaMalloc failure");

        cudaMemcpy(d_arg, arg, sizeof(struct aux_args), cudaMemcpyHostToDevice);
        cudaMemcpy(d_calc, calc_local, sizeof(int) * this_cnt, cudaMemcpyHostToDevice);
        cudaERRO("cudaMemcpy H2D failure\n");

        gpu_make_iterations<<<256, arg->threads>>>(d_arg, d_calc, this_cnt);
        cudaERRO("Kernel launch failure\n");

        /* Wait for all blocks/threads finishes their work */
        cudaDeviceSynchronize();
        cudaERRO("cudaDeviceSynchronize failure\n");

        cudaMemcpy(calc_local, d_calc, this_cnt*sizeof(int), cudaMemcpyDeviceToHost);
        cudaERRO("cudaMemcpy D2H failure\n");

        cudaFree(d_calc);
        cudaFree(d_arg);
    }

    /* Recovering data from mpi-processes */
    MPI_Gatherv (calc_local, this_cnt, MPI_INT,
        calc, cnt, starts, MPI_INT,
        root, MPI_COMM_WORLD);

    if (world_rank == 0) {
        generate_image (arg, calc);

        /* Let it go, let it go */
        free(calc);
    }

    free(calc_local);
    free(cnt);
    free(starts);

    /* Finalize the MPI environment */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}

/* https://stackoverflow.com/questions/9269399/sending-blocks-of-2d-array-in-c-using-mpi/9271753#9271753 */
