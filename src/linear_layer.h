#include "ndarray.h"
#include "variable.h"

#ifndef TEENY_AUTOGRAD_C_LINEAR_LAYER_H
#define TEENY_AUTOGRAD_C_LINEAR_LAYER_H

typedef struct linear_layer
{
    int in_size;
    int out_size;
    variable *weights;
    variable *bias;
} linear_layer;

linear_layer *new_linear_layer(int in_size, int out_size);

variable *forward_linear_layer(linear_layer *ll, variable *input);

void free_linear_layer(linear_layer *ll);

#endif