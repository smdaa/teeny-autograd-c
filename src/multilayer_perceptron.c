#include <stdlib.h>
#include "multilayer_perceptron.h"

multilayer_perceptron *new_multilayer_perceptron(int n_layers, int batch_size, int *in_sizes, int *out_sizes)
{
    ndarray *place_holder;
    multilayer_perceptron *mlp = (multilayer_perceptron *)malloc(sizeof(multilayer_perceptron));
    mlp->n_layers = n_layers;
    mlp->batch_size = batch_size;
    mlp->in_sizes = (int *)malloc(n_layers * sizeof(int));
    mlp->out_sizes = (int *)malloc(n_layers * sizeof(int));
    mlp->weights = (variable **)malloc(n_layers * sizeof(variable *));
    mlp->bias = (variable **)malloc(n_layers * sizeof(variable *));
    for (int i = 0; i < n_layers; i++)
    {
        mlp->in_sizes[i] = in_sizes[i];
        mlp->out_sizes[i] = out_sizes[i];

        place_holder = random_ndrray(2, (int[]){in_sizes[i], out_sizes[i]});
        mlp->weights[i] = new_variable(place_holder);
        free_ndarray(&place_holder);

        place_holder = random_ndrray(2, (int[]){1, out_sizes[i]});
        mlp->bias[i] = new_variable(place_holder);
        free_ndarray(&place_holder);
    }

    return mlp;
}

variable *forward_multilayer_perceptron(multilayer_perceptron *mlp, variable *input)
{
    ndarray *place_holder = ones_ndarray(2, (int[]){mlp->batch_size, 1});
    variable *output = input;
    for (int i = 0; i < mlp->n_layers; i++)
    {
        output = matmul_variable(output, mlp->weights[i]);
        output = add_variable(output, matmul_variable(new_variable(place_holder), mlp->bias[i]));
    }
    free_ndarray(&place_holder);

    return output;
}

void free_multilayer_perceptron(multilayer_perceptron **mlp)
{
    if (*mlp == NULL)
    {
        return;
    }

    if ((*mlp)->in_sizes != NULL)
    {
        free((*mlp)->in_sizes);
        (*mlp)->in_sizes = NULL;
    }

    if ((*mlp)->out_sizes != NULL)
    {
        free((*mlp)->out_sizes);
        (*mlp)->out_sizes = NULL;
    }

    if ((*mlp)->weights != NULL)
    {
        /*
        for (int i = 0; i < (*mlp)->n_layers; i++)
        {
            free_variable(&((*mlp)->weights[i]));
        }
        */
        free((*mlp)->weights);
        (*mlp)->weights = NULL;
    }

    if ((*mlp)->bias != NULL)
    {
        /*
        for (int i = 0; i < (*mlp)->n_layers; i++)
        {
            free_variable(&((*mlp)->bias[i]));
        }
        */
        free((*mlp)->bias);
        (*mlp)->bias = NULL;
    }

    free(*mlp);
    *mlp = NULL;
}
