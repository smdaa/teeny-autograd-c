#include <stdlib.h>
#include "linear_layer.h"

linear_layer *new_linear_layer(int in_size, int out_size)
{
    linear_layer *ll = (linear_layer *)malloc(sizeof(linear_layer));
    ll->in_size = in_size;
    ll->out_size = out_size;
    ll->weights = new_variable(random_ndrray(2, (int[]){in_size, out_size}));
    ll->bias = new_variable(random_ndrray(1, (int[]){out_size}));

    return ll;
}

variable *forward_linear_layer(linear_layer *ll, variable *input)
{
    return add_variable(matmul_variable(input, ll->weights), ll->bias);
}

void free_linear_layer(linear_layer *ll)
{
    free_variable(ll->weights);
    free_variable(ll->bias);
}