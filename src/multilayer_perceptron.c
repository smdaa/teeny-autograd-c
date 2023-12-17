#include "multilayer_perceptron.h"

#include <stdlib.h>

multilayer_perceptron *
new_multilayer_perceptron(int n_layers, int batch_size, int *in_sizes,
                          int *out_sizes, activation_function *activations,
                          random_initialisation *random_initialisations) {
  ndarray *place_holder;
  multilayer_perceptron *mlp =
      (multilayer_perceptron *)malloc(sizeof(multilayer_perceptron));
  mlp->n_layers = n_layers;
  mlp->batch_size = batch_size;
  mlp->in_sizes = (int *)malloc(n_layers * sizeof(int));
  mlp->out_sizes = (int *)malloc(n_layers * sizeof(int));
  mlp->weights = (variable **)malloc(n_layers * sizeof(variable *));
  mlp->bias = (variable **)malloc(n_layers * sizeof(variable *));
  mlp->activations =
      (activation_function *)malloc(n_layers * sizeof(activation_function));
  mlp->random_initialisations =
      (random_initialisation *)malloc(n_layers * sizeof(random_initialisation));
  for (int i = 0; i < n_layers; i++) {
    mlp->in_sizes[i] = in_sizes[i];
    mlp->out_sizes[i] = out_sizes[i];
    mlp->activations[i] = activations[i];
    mlp->random_initialisations[i] = random_initialisations[i];

    switch (mlp->random_initialisations[i]) {
    case UNIFORM:
      place_holder = random_ndrray(2, (int[]){in_sizes[i], out_sizes[i]});
      mlp->weights[i] = new_variable(place_holder);
      free_ndarray(&place_holder);

      place_holder = random_ndrray(2, (int[]){1, out_sizes[i]});
      mlp->bias[i] = new_variable(place_holder);
      free_ndarray(&place_holder);
      break;
    case TRUNCATED_NORMAL:
      place_holder = random_truncated_ndarray(
          2, (int[]){in_sizes[i], out_sizes[i]}, 0.0, 1.0, -2.0, 2.0);
      mlp->weights[i] = new_variable(place_holder);
      free_ndarray(&place_holder);

      place_holder = random_truncated_ndarray(2, (int[]){1, out_sizes[i]}, 0.0,
                                              1.0, -2.0, 2.0);
      mlp->bias[i] = new_variable(place_holder);
      free_ndarray(&place_holder);
      break;

    default:
      break;
    }
  }

  return mlp;
}

variable *forward_multilayer_perceptron(multilayer_perceptron *mlp,
                                        variable *input) {
  variable *output = input;
  for (int i = 0; i < mlp->n_layers; i++) {
    output = matmul_variable(output, mlp->weights[i]);
    output = add_variable(output, mlp->bias[i]);
    switch (mlp->activations[i]) {
    case LINEAR:
      break;
    case RELU:
      output = relu_variable(output);
      break;
    case SIGMOID:
      output = sigmoid_variable(output);
      break;
    case SOFTMAX:
      output = softmax_variable(output, 1);
      break;
    case TANH:
      output = tanh_variable(output);
      break;
    default:
      break;
    }
  }

  return output;
}

void free_multilayer_perceptron(multilayer_perceptron **mlp) {
  if (*mlp == NULL) {
    return;
  }

  if ((*mlp)->in_sizes != NULL) {
    free((*mlp)->in_sizes);
    (*mlp)->in_sizes = NULL;
  }

  if ((*mlp)->out_sizes != NULL) {
    free((*mlp)->out_sizes);
    (*mlp)->out_sizes = NULL;
  }

  if ((*mlp)->weights != NULL) {
    free((*mlp)->weights);
    (*mlp)->weights = NULL;
  }

  if ((*mlp)->bias != NULL) {
    free((*mlp)->bias);
    (*mlp)->bias = NULL;
  }
  if ((*mlp)->activations != NULL) {
    free((*mlp)->activations);
    (*mlp)->activations = NULL;
  }
  if ((*mlp)->random_initialisations != NULL) {
    free((*mlp)->random_initialisations);
    (*mlp)->random_initialisations = NULL;
  }

  free(*mlp);
  *mlp = NULL;
}
