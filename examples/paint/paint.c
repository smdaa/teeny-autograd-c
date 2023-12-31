#include "../../extern/lodepng/lodepng.h"
#include "../../src/multilayer_perceptron.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

void create_image(const char *filename, int width, int height,
                  unsigned char *image_data) {
  if (lodepng_encode24_file(filename, image_data, width, height) != 0) {
    printf("Error encoding PNG file\n");
    exit(1);
  }
}

int main(void) {
  int layer_size = 32;
  int batch_size = 64;
  int height = 32;
  int width = 32;
  NDARRAY_TYPE zoom = 1.0;

  // Create model
  multilayer_perceptron *mlp = new_multilayer_perceptron(
      9, batch_size,
      (int[]){3, layer_size, layer_size, layer_size, layer_size, layer_size,
              layer_size, layer_size, layer_size},
      (int[]){layer_size, layer_size, layer_size, layer_size, layer_size,
              layer_size, layer_size, layer_size, 3},
      (activation_function[]){TANH, TANH, TANH, TANH, TANH, TANH, TANH, TANH,
                              SIGMOID},
      (random_initialisation[]){NORMAL, NORMAL, NORMAL, NORMAL, NORMAL, NORMAL,
                                NORMAL, NORMAL, NORMAL});

  // Generate data
  ndarray *x = zeros_ndarray(2, (int[]){height * width, 3});
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      x->data[3 * (i * width + j)] = ((NDARRAY_TYPE)i / height - 0.5) * zoom;
      x->data[3 * (i * width + j) + 1] = ((NDARRAY_TYPE)j / width - 0.5) * zoom;
      x->data[3 * (i * width + j) + 2] =
          sqrt(pow((NDARRAY_TYPE)i / height - 0.5, 2.0) +
               pow((NDARRAY_TYPE)j / width - 0.5, 2.0)) *
          zoom;
    }
  }

  // Generate batches
  int n_batches = (x->shape[0] + batch_size - 1) / batch_size;
  variable **x_batches = create_batches(x, batch_size);
  variable **y_batches = (variable **)malloc(n_batches * sizeof(variable *));
  free_ndarray(&x);

  // Forward pass
  for (int i = 0; i < n_batches; i++) {
    variable *x_batch = (x_batches[i]);
    variable *y_batch = forward_batch_multilayer_perceptron(mlp, x_batch);
    y_batches[i] = y_batch;
  }

  // Create an image
  // Create an image
  unsigned char *image_data =
      (unsigned char *)malloc(3 * width * height * sizeof(unsigned char));
  int dataIndex = 0;
  for (int b = 0; b < n_batches; b++) {
    variable *y_batch = y_batches[b];
    for (int j = 0; j < y_batch->val->shape[0]; j++) {
      for (int k = 0; k < 3; k++) {
        image_data[dataIndex++] =
            (unsigned char)(255 * y_batch->val->data[3 * j + k]);
      }
    }
  }
  for (int i = 0; i < n_batches; i++) {
    free_graph_variable(&y_batches[i]);
  }
  free(y_batches);
  free(x_batches);
  free_multilayer_perceptron(&mlp);

  // Write the image to a file using LodePNG
  create_image("output.png", width, height, image_data);

  free(image_data);

  return 0;
}