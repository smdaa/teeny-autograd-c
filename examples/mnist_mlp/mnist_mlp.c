#include "../../src/multilayer_perceptron.h"
#include <stdlib.h>
#include <assert.h>

variable **create_batches(ndarray *x, int batch_size) {
  int n_batches = (x->shape[0] + batch_size - 1) / batch_size;
  variable **batches = (variable **)malloc(n_batches * sizeof(variable *));

  for (int i = 0; i < n_batches; i++) {
    int start = i * batch_size;
    int end = start + batch_size;
    if (end > x->shape[0])
      end = x->shape[0];
    ndarray *batch_data = zeros_ndarray(2, (int[]){end - start, x->shape[1]});
    for (int j = start; j < end; j++) {
      for (int k = 0; k < x->shape[1]; k++) {
        batch_data->data[(j - start) * x->shape[1] + k] =
            x->data[j * x->shape[1] + k];
      }
    }
    batches[i] = new_variable(batch_data);
    free_ndarray(&batch_data);
  }

  return batches;
}

variable *cross_entropy_loss(variable *logits, variable *y) {
  variable *y_hat_exp = exp_variable(logits);
  variable *y_hat_sum = sum_variable(y_hat_exp, 1);
  ndarray *temp = full_ndarray(y_hat_sum->val->dim, y_hat_sum->val->shape,
                               NDARRAY_TYPE_EPSILON);
  variable *y_hat_log_sum =
      log_variable(add_variable(y_hat_sum, new_variable(temp)));
  free_ndarray(&temp);

  variable *y_hat_softmax = subtract_variable(logits, y_hat_log_sum);

  variable *product = multiply_variable(y, y_hat_softmax);
  variable *neg_product = negate_variable(product);
  variable *loss = sum_variable(neg_product, 1);

  free_ndarray(&(loss->grad));
  loss->grad = ones_ndarray(loss->val->dim, loss->val->shape);
  return loss;
}

int main(void) {
  const char *dataDir = getenv("MNIST_MLP_DATA_DIR");
  if (dataDir == NULL) {
    fprintf(stderr,
            "Error: MNIST_MLP_DATA_DIR environment variable not set.\n");
    return 1;
  }

  int n_layers = 4;
  int in_sizes[] = {28 * 28, 64, 32, 16};
  int out_sizes[] = {64, 32, 16, 10};
  int batch_size = 64;
  int n_epochs = 100;
  NDARRAY_TYPE learning_rate = 0.01;

  // Load training data
  char path[256];
  snprintf(path, sizeof(path), "%s/x_train.txt", dataDir);
  ndarray *x_train = read_ndarray(path);
  snprintf(path, sizeof(path), "%s/y_train.txt", dataDir);
  ndarray *y_train = read_ndarray(path);
  assert(x_train->shape[0] == y_train->shape[0]);
  int n_batches = (x_train->shape[0] + batch_size - 1) / batch_size;
  variable **x_batches = create_batches(x_train, batch_size);
  variable **y_batches = create_batches(y_train, batch_size);

  free_ndarray(&x_train);
  free_ndarray(&y_train);

  // Create model
  activation_function activations[] = {SIGMOID, SIGMOID, SIGMOID, LINEAR};
  random_initialisation random_initialisations[] = {UNIFORM, UNIFORM, UNIFORM,
                                                    UNIFORM};
  multilayer_perceptron *mlp =
      new_multilayer_perceptron(n_layers, batch_size, in_sizes, out_sizes,
                                activations, random_initialisations);

  // Train model
  train_multilayer_perceptron(mlp, x_batches, y_batches, n_batches, n_epochs,
                              learning_rate, cross_entropy_loss);

  // House keeping
  for (int i = 0; i < n_batches; i++) {
    free_graph_variable(&x_batches[i]);
    free_graph_variable(&y_batches[i]);
  }
  free(x_batches);
  free(y_batches);
  for (int i = 0; i < n_layers; i++) {
    free_graph_variable(&mlp->weights[i]);
    free_graph_variable(&mlp->bias[i]);
  }
  free_multilayer_perceptron(&mlp);

  printf("Done.\n");
}