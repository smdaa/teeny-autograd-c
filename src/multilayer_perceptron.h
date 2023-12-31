#include "variable.h"

#ifndef TEENY_AUTOGRAD_C_MULTILAYER_PERCEPTRON_H
#define TEENY_AUTOGRAD_C_MULTILAYER_PERCEPTRON_H

typedef enum {
  UNIFORM,
  NORMAL,
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
  variable **weights_copy;
  variable **bias_copy;
  activation_function *activations;
  random_initialisation *random_initialisations;

} multilayer_perceptron;

multilayer_perceptron *
new_multilayer_perceptron(int n_layers, int batch_size, int *in_sizes,
                          int *out_sizes, activation_function *activations,
                          random_initialisation *random_initialisations);

variable *forward_batch_multilayer_perceptron(multilayer_perceptron *mlp,
                                              variable *x_batch);

void train_multilayer_perceptron(multilayer_perceptron *mlp,
                                 variable **x_batches, variable **y_batches,
                                 int n_batches, int n_epochs,
                                 NDARRAY_TYPE learning_rate,
                                 variable *(*loss_fn)(variable *, variable *));

void zero_grad_multilayer_perceptron(multilayer_perceptron *mlp);

void update_multilayer_perceptron(multilayer_perceptron *mlp,
                                  NDARRAY_TYPE learning_rate);

void free_multilayer_perceptron(multilayer_perceptron **mlp);

#endif
