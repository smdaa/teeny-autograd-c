#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../extern/libpng/png.h"
#include "../../src/multilayer_perceptron.h"

variable **create_batches(ndarray *x, int batch_size) {
    int n_batches = (x->shape[0] + batch_size - 1) / batch_size;
    variable **batches = (variable **) malloc(n_batches * sizeof(variable *));

    for (int i = 0; i < n_batches; i++) {
        int start = i * batch_size;
        int end = start + batch_size;
        if (end > x->shape[0])
            end = x->shape[0];
        ndarray *batch_data = zeros_ndarray(2, (int[]) {end - start, x->shape[1]});
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

int main(void) {
    int layer_size = 32;
    int batch_size = 64;
    int height = 1024;
    int width = 1024;
    NDARRAY_TYPE zoom = 1.0;

    // Create model
    multilayer_perceptron *mlp = new_multilayer_perceptron(
            9, batch_size,
            (int[]) {3, layer_size, layer_size, layer_size, layer_size, layer_size, layer_size, layer_size, layer_size},
            (int[]) {layer_size, layer_size, layer_size, layer_size, layer_size, layer_size, layer_size, layer_size, 3},
            (activation_function[]) {TANH, TANH, TANH, TANH, TANH, TANH, TANH, TANH, SIGMOID},
            (random_initialisation[]) {NORMAL, NORMAL, NORMAL, NORMAL, NORMAL, NORMAL, NORMAL, NORMAL, NORMAL});

    // Generate data
    ndarray *x = zeros_ndarray(2, (int[]) {height * width, 3});
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            x->data[3 * (i * width + j)] = ((NDARRAY_TYPE) i / height - 0.5) * zoom;
            x->data[3 * (i * width + j) + 1] = ((NDARRAY_TYPE) j / width - 0.5) * zoom;
            x->data[3 * (i * width + j) + 2] =
                    sqrt(pow((NDARRAY_TYPE) i / height - 0.5, 2.0) +
                         pow((NDARRAY_TYPE) j / width - 0.5, 2.0)) *
                    zoom;
        }
    }

    // Generate batches
    int n_batches = (x->shape[0] + batch_size - 1) / batch_size;
    variable **x_batches = create_batches(x, batch_size);
    variable **y_batches = (variable **) malloc(n_batches * sizeof(variable *));
    free_ndarray(&x);

    // Forward pass
    for (int i = 0; i < n_batches; i++) {
        variable *x_batch = (x_batches[i]);
        variable *y_batch = forward_multilayer_perceptron(mlp, x_batch);
        y_batches[i] = y_batch;
    }

    // Create an image
    png_bytep *row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);
    for (int i = 0; i < height; i++) {
        row_pointers[i] = (png_byte *) malloc(3 * width * sizeof(png_byte));
    }
    int dataIndex = 0;
    for (int b = 0; b < n_batches; b++) {
        variable *y_batch = y_batches[b];
        for (int j = 0; j < y_batch->val->shape[0]; j++) {
            int row = dataIndex / width;
            int col = dataIndex % width;
            for (int k = 0; k < 3; k++) {
                row_pointers[row][3 * col + k] =
                        (png_byte) (255 * y_batch->val->data[3 * j + k]);
            }
            dataIndex++;
        }
    }
    for (int i = 0; i < n_batches; i++) {
        free_graph_variable(&y_batches[i]);
    }
    free(y_batches);
    free(x_batches);
    free_multilayer_perceptron(&mlp);

    // Write the image to a file
    FILE *fp = fopen("output.png", "wb");
    if (!fp) {
        printf("Cannot open file for writing\n");
        return 1;
    }

    png_structp png_ptr =
            png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        printf("Cannot create write struct\n");
        return 1;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, (png_infopp) NULL);
        printf("Cannot create info struct\n");
        return 1;
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

    for (int i = 0; i < height; i++) {
        free(row_pointers[i]);
    }
    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);

    return 0;
}