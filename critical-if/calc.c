#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Global values
double vals[50];
double mean;
double variance;
double ci;

double mean_calc() {
    double mean = 0.0;

    for (int i = 0; i < 50; i++)
        mean = mean + (vals[i] / 50);

    return mean;
}

double variance_calc() {
    double var = 0.0;

    for (int i = 0; i < 50; i++)
        var = var + (pow((vals[i] - mean), 2.0) / (49));

    return var;
}

double ci_calc() {
  double ci;
  double t_score;

  // Hardcoded T_student value
  t_score = 2.009574;
  ci = t_score * sqrt(variance / 50);

  return ci;
}


void main() {

    // gcc calc.c -lm -o calc

    // Reading in.dat
    FILE *file;
    file = fopen("in.dat", "r");

    int count = 0;
    while (!feof(file)) {
        fscanf(file, "%lf", &(vals[count++]));
    }
    fclose(file);

    mean = mean_calc();
    variance = variance_calc();
    ci = ci_calc();

    printf("%lfs %lfs %lfs %lfs\n", mean, ci, (mean-ci), (mean+ci));
}
