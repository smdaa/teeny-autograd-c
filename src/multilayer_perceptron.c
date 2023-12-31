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
      NDARRAY_TYPE stdv = 1.0 / sqrt((NDARRAY_TYPE)mlp->out_sizes[i]);

      place_holder = random_uniform_ndarray(
          2, (int[]){in_sizes[i], out_sizes[i]}, -stdv, stdv);
      mlp->weights[i] = new_variable(place_holder);
      free_ndarray(&place_holder);

      place_holder =
          random_uniform_ndarray(2, (int[]){1, out_sizes[i]}, -stdv, stdv);
      mlp->bias[i] = new_variable(place_holder);
      free_ndarray(&place_holder);
      break;
    case NORMAL:
      place_holder = random_normal_ndarray(
          2, (int[]){in_sizes[i], out_sizes[i]}, 0.0, 1.0);
      mlp->weights[i] = new_variable(place_holder);
      free_ndarray(&place_holder);

      place_holder =
          random_normal_ndarray(2, (int[]){1, out_sizes[i]}, 0.0, 1.0);
      mlp->bias[i] = new_variable(place_holder);
      free_ndarray(&place_holder);
      break;
    case TRUNCATED_NORMAL:
      place_holder = random_truncated_normal_ndarray(
          2, (int[]){in_sizes[i], out_sizes[i]}, 0.0, 1.0, -2.0, 2.0);
      mlp->weights[i] = new_variable(place_holder);
      free_ndarray(&place_holder);

      place_holder = random_truncated_normal_ndarray(
          2, (int[]){1, out_sizes[i]}, 0.0, 1.0, -2.0, 2.0);
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

void zero_grad_multilayer_perceptron(multilayer_perceptron *mlp) {
  ndarray *place_holder;
  for (int i = 0; i < mlp->n_layers; i++) {
    place_holder = mlp->weights[i]->grad;
    mlp->weights[i]->grad =
        zeros_ndarray(mlp->weights[i]->grad->dim, mlp->weights[i]->grad->shape);
    free_ndarray(&place_holder);

    place_holder = mlp->bias[i]->grad;
    mlp->bias[i]->grad =
        zeros_ndarray(mlp->bias[i]->grad->dim, mlp->bias[i]->grad->shape);
    free_ndarray(&place_holder);
  }
}

void update_multilayer_perceptron(multilayer_perceptron *mlp,
                                  NDARRAY_TYPE learning_rate) {
  ndarray *place_holder;
  ndarray *temp;
  for (int i = 0; i < mlp->n_layers; i++) {
    place_holder = mlp->weights[i]->val;
    temp = multiply_ndarray_scalar(mlp->weights[i]->grad, learning_rate);
    mlp->weights[i]->val = subtract_ndarray_ndarray(mlp->weights[i]->val, temp);
    free_ndarray(&place_holder);
    free_ndarray(&temp);

    place_holder = mlp->bias[i]->val;
    temp = multiply_ndarray_scalar(mlp->bias[i]->grad, learning_rate);
    mlp->bias[i]->val = subtract_ndarray_ndarray(mlp->bias[i]->val, temp);

    free_ndarray(&place_holder);
    free_ndarray(&temp);
  }
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
