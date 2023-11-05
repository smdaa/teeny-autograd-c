#include "variable.h"

#ifndef TEENY_AUTOGRAD_C_MULTILAYER_PERCEPTRON_H
#define TEENY_AUTOGRAD_C_MULTILAYER_PERCEPTRON_H

typedef struct multilayer_perceptron
{
    int n_layers;
    int batch_size;
    int *in_sizes;
    int *out_sizes;
    variable **weights;
    variable **bias;

} multilayer_perceptron;

multilayer_perceptron *new_multilayer_perceptron(int n_layers, int batch_size, int *in_sizes, int *out_sizes);

variable* forward_multilayer_perceptron(multilayer_perceptron *mlp, variable* input);

void free_multilayer_perceptron(multilayer_perceptron **mlp);
#endif
