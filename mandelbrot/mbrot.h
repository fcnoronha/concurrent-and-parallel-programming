#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <png.h> /* libpng */

/* Store all parameters passed as argument to main */
struct aux_args {
    float c0_real; /* Real part of c0 */
    float c0_imag; /* Imaginary part of c0 */
    float c1_real; /* Real part of c1 */
    float c1_imag; /* Imaginary part of c1 */
    unsigned w; /* width, in pixels, of the image to be generated */
    unsigned h; /* high, in pixels, of the image to be generated */
    unsigned is_cpu; /* True if the programs needs to run in the cpu */
    unsigned threads; /* Number of threads/blocks to be executed */
    const char *saida; /* path to output image */
};

/* Create a png image from a int array calc[], what is an 1D
   representation of a 2D array, where each position is a certain
   number of iterations */
void generate_image(struct aux_args *arg, int *calc);

/* Iterate using maldelbrot rule in each potion */
unsigned cpu_iterate (float c_real, float c_imag);

/* Receive input arguments in arg, an array calc, of size WxH, put the number
   of iterations corresponding to each pixel of image */
void cpu_make_iterations (struct aux_args *arg, int *calc);

/* Auxiliar function for complex module */
inline float cpu_squared_moduleZ (float z_real, float z_imag);

/* Receive input arguments in arg, an array calc, of size WxH, put the number
   of iterations corresponding to each pixel of image */
__global__
void gpu_make_iterations (struct aux_args *arg, int *calc);

/* Iterate using maldelbrot rule in each potion */
__device__
unsigned gpu_iterate (float c_real, float c_imag);

/* Auxiliar function for complex module */
__device__
inline float gpu_squared_moduleZ (float z_real, float z_imag);
