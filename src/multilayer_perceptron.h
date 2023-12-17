#include "variable.h"

#ifndef TEENY_AUTOGRAD_C_MULTILAYER_PERCEPTRON_H
#define TEENY_AUTOGRAD_C_MULTILAYER_PERCEPTRON_H

typedef enum {
  UNIFORM,
  TRUNCATED_NORMAL,
} random_initialisation;

typedef enum {
  LINEAR,
  RELU,
  SIGMOID,
  SOFTMAX,
  TANH,
} activation_function;

typedef struct multilayer_perceptron {
  int n_layers;
  int batch_size;
  int *in_sizes;
  int *out_sizes;
  variable **weights;
  variable **bias;
  activation_function *activations;
  random_initialisation *random_initialisations;

} multilayer_perceptron;

multilayer_perceptron *
new_multilayer_perceptron(int n_layers, int batch_size, int *in_sizes,
                          int *out_sizes, activation_function *activations,
                          random_initialisation *random_initialisations);

variable *forward_multilayer_perceptron(multilayer_perceptron *mlp,
                                        variable *input);

void free_multilayer_perceptron(multilayer_perceptron **mlp);
#endif
