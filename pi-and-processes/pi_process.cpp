//#include <sys/types.h>
#include <sys/mman.h>
//#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <unistd.h>
//#include <string.h>
#include <fcntl.h>
#include <math.h>
#include <vector>
#include <iostream>

/*
*   Program to aproximate PI using Riemann integral of the function
*   f(x) = sqrt(1 - x²), with x e [0,1].
*
*   Execute specifing number of processes in the CPU you want to use and the
*   number of points for the aproximation. Run as follows:
*   ./pi_process <NUM_PROCESSES> <NUM_POINTS>
*
*   By Felipe Noronha at IME-USP 2019
*/

// For testing
#define dbg(x) std::cout << #x << " = " << x << "\n"

// Error handling
#define ERRO(msg, name) { \
        fprintf(stderr, msg); \
        shm_unlink(name); \
        exit(EXIT_FAILURE); \
}

// Function to calculate calculate the integral, using Riemann method, in
// a given interval
long double riemann_integration(unsigned int a, unsigned int b, unsigned int num_points) {

    // Local integration variable
    long double ret = 0;
    // Sectors of circle
    long double interval_size = 1.0 / num_points;

    // Integrates f(x) = sqrt(1 - x^2) in [t->start, t->end[
    for (int i = a; i < b; ++i) {
        long double x = (i * interval_size) + interval_size / 2;
        ret += sqrt(1 - (x * x)) * interval_size;
    }

    ret *= 4.0;
    return ret;
}

int main(int argc, char const *argv[]) {

    // Used for testing. If true, program will print absolute difference between
    // the real pi number and the calculated one and will also print the elapsed
    // time.
    bool test_flag = false;
    struct timeval start, end;

    // Pasing input
    unsigned int num_processes, num_points;
    sscanf(argv[1], "%u", &num_processes);
    sscanf(argv[2], "%u", &num_points);

    // Shared memory variables
    long double* shared_memory;
    int shm_fd;
    int msize;
    const char *name = "RIEMANN_INTEGRATION";

    // Size of the array in the shared memory
    msize = (num_processes)*sizeof(long double);

    // Open the memory with flags to create the segment, check if a segment
    // with same name already exists and read-write access.
    shm_fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRWXU|S_IRWXG);
    if (shm_fd < 0)
        ERRO("Error opening shared memory!\n", name);

    // Giving size to this memory segment
    ftruncate(shm_fd, msize);

    // Allocating the shared memory with option to be shared
    shared_memory = (long double *)mmap(NULL, msize, PROT_READ|PROT_WRITE,
        MAP_SHARED, shm_fd, 0);
    if (shared_memory == NULL)
        ERRO("Error while mmap()ing", name);

    // Used to manage creation and created processes
    pid_t aux_pid;
    std::vector<pid_t> initialized_pids;

    // Time in the start of processing
    gettimeofday(&start, NULL);

    unsigned int processess_with_more_interval = num_points % num_processes;
    for (unsigned int i = 0; i < num_processes; ++i) {

        // This will only be called from the parent process
        aux_pid = fork();

        if (aux_pid < 0) {
            ERRO("Error while forking!\n", name);
        }

        // Child process
        else if (aux_pid == 0) {

            // Defining interval that child process will take care of
            unsigned int interval = num_points / num_processes;
            if (i < processess_with_more_interval)
                    interval += 1;

            // [a, b(
            unsigned int a = i * interval;
            unsigned int b = (i + 1) * interval;

            // dbg(interval);
            shared_memory[i] = riemann_integration(a, b, num_points);

            exit(EXIT_SUCCESS);
        }

        // Parent process
        else
            initialized_pids.push_back(aux_pid);
    }

    // Waiting for all child processes to exit
    for (pid_t pid : initialized_pids)
        waitpid(pid, NULL, 0);

    // Final answer
    long double pi = 0.0;
    for (int i = 0; i < num_processes; ++i)
        pi += shared_memory[i];

    // Final time
    gettimeofday(&end, NULL);

    if (test_flag) {

        double elapsed_time = (end.tv_sec - start.tv_sec) +
                (end.tv_usec - start.tv_usec) / 1000000.0;

        long double r_pi = acos(-1);
        long double abs_err = r_pi - pi;
        if (abs_err < 0) abs_err = -abs_err;

        printf("[[%1.20Lf], [%.4f]]\n", abs_err, elapsed_time);
    }

    else {
        printf("%1.20Lf\n", pi);
    }

    // Closing shared files
    shm_unlink(name);
    munmap(shared_memory, msize);

    return 0;
}
