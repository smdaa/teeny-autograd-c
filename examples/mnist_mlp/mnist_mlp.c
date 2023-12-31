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

  int n_layers = 2;
  int in_sizes[] = {28 * 28, 16};
  int out_sizes[] = {16, 10};
  int batch_size = 256;
  int n_epochs = 2;
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
  for (int i = 0; i < n_batches; i++) {
    x_batches[i]->tag = DO_NOT_FREE;
    y_batches[i]->tag = DO_NOT_FREE;
  }

  free_ndarray(&x_train);
  free_ndarray(&y_train);

  // Create model
  activation_function activations[] = {SIGMOID, LINEAR};
  random_initialisation random_initialisations[] = {UNIFORM, UNIFORM};
  multilayer_perceptron *mlp =
      new_multilayer_perceptron(n_layers, batch_size, in_sizes, out_sizes,
                                activations, random_initialisations);
  for (int i = 0; i < mlp->n_layers; i++) {
    mlp->weights[i]->tag = DO_NOT_FREE;
    mlp->bias[i]->tag = DO_NOT_FREE;
  }

  // Train model
  for (int i = 0; i < n_epochs; i++) {
    for (int j = 0; j < n_batches; j++) {
      variable *x_batch = x_batches[j];
      variable *y_batch = y_batches[j];
      variable *y_hat_batch = forward_multilayer_perceptron(mlp, x_batch);
      variable *loss_batch = cross_entropy_loss(y_hat_batch, y_batch);

      zero_grad_multilayer_perceptron(mlp);
      backward_variable(loss_batch);
      update_multilayer_perceptron(mlp, learning_rate);

      if (j % 100 == 0) {
        printf("Epoch %d, batch %d, loss: %f\n", i, j,
               sum_all_ndarray(loss_batch->val));
      }

      if (i == n_epochs - 1) {
        x_batch->tag = OK_TO_FREE;
        y_batch->tag = OK_TO_FREE;
        x_batch->ref_count = 0;
        y_batch->ref_count = 0;
        if (j == n_batches - 1) {
          for (int k = 0; k < n_layers; k++) {
            mlp->weights[k]->tag = OK_TO_FREE;
            mlp->bias[k]->tag = OK_TO_FREE;
            mlp->weights[k]->ref_count = 0;
            mlp->bias[k]->ref_count = 0;
          }
        }
      }

      free_graph_variable(&loss_batch);
    }
  }

  free(x_batches);
  free(y_batches);
  free_multilayer_perceptron(&mlp);

  printf("Done.\n");
}