#include "variable.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

NDARRAY_TYPE relu(NDARRAY_TYPE a) { return fmax(a, 0.0); }

NDARRAY_TYPE sigmoid(NDARRAY_TYPE a) { return 1.0 / (1.0 + exp(-a)); }

NDARRAY_TYPE d_sigmoid(NDARRAY_TYPE a) {
  return exp(-a) / pow(1.0 + exp(-a), 2.0);
}

NDARRAY_TYPE tanh(NDARRAY_TYPE a) {
  return (exp(a) - exp(-a)) / (exp(a) + exp(-a));
}

NDARRAY_TYPE d_tanh(NDARRAY_TYPE a) {
  return 1.0 - pow((exp(a) - exp(-a)) / (exp(a) + exp(-a)), 2.0);
}

variable *new_variable(ndarray *val) {
  variable *var = (variable *)malloc(sizeof(variable));
  var->val = copy_ndarray(val);
  var->grad = zeros_ndarray(val->dim, val->shape);
  var->children = NULL;
  var->n_children = 0;
  var->backward = NULL;
  var->ref_count = 0;

  return var;
}

void add_backward(variable *var) {
  ndarray *place_holder;
  ndarray *reduced_grad;
  ndarray *temp;

  place_holder = var->children[0]->grad;
  reduced_grad = copy_ndarray(var->grad);
  for (int i = 0; i < var->children[0]->val->dim; i++) {
    if (var->children[0]->val->shape[i] != var->grad->shape[i]) {
      temp = sum_ndarray(reduced_grad, i);
      free_ndarray(&reduced_grad);
      reduced_grad = temp;
    }
  }
  var->children[0]->grad =
      add_ndarray_ndarray(reduced_grad, var->children[0]->grad);
  free_ndarray(&place_holder);
  free_ndarray(&reduced_grad);

  place_holder = var->children[1]->grad;
  reduced_grad = copy_ndarray(var->grad);
  for (int i = 0; i < var->children[1]->val->dim; i++) {
    if (var->children[1]->val->shape[i] != var->grad->shape[i]) {
      temp = sum_ndarray(reduced_grad, i);
      free_ndarray(&reduced_grad);
      reduced_grad = temp;
    }
  }
  var->children[1]->grad =
      add_ndarray_ndarray(reduced_grad, var->children[1]->grad);
  free_ndarray(&place_holder);
  free_ndarray(&reduced_grad);
}

void subtract_backward(variable *var) {
  ndarray *place_holder;
  ndarray *reduced_grad;
  ndarray *temp;

  place_holder = var->children[0]->grad;
  reduced_grad = copy_ndarray(var->grad);
  for (int i = 0; i < var->children[0]->val->dim; i++) {
    if (var->children[0]->val->shape[i] != var->grad->shape[i]) {
      temp = sum_ndarray(reduced_grad, i);
      free_ndarray(&reduced_grad);
      reduced_grad = temp;
    }
  }
  var->children[0]->grad =
      add_ndarray_ndarray(reduced_grad, var->children[0]->grad);
  free_ndarray(&place_holder);
  free_ndarray(&reduced_grad);

  place_holder = var->children[1]->grad;
  reduced_grad = copy_ndarray(var->grad);
  for (int i = 0; i < var->children[1]->val->dim; i++) {
    if (var->children[1]->val->shape[i] != var->grad->shape[i]) {
      temp = sum_ndarray(reduced_grad, i);
      free_ndarray(&reduced_grad);
      reduced_grad = temp;
    }
  }
  var->children[1]->grad =
      subtract_ndarray_ndarray(var->children[1]->grad, reduced_grad);
  free_ndarray(&place_holder);
  free_ndarray(&reduced_grad);
}

void multiply_backward(variable *var) {
  ndarray *place_holder;
  ndarray *reduced_grad;
  ndarray *temp;

  place_holder = var->children[0]->grad;
  temp = multiply_ndarray_ndarray(var->children[1]->val, var->grad);
  reduced_grad = copy_ndarray(temp);
  for (int i = 0; i < var->children[0]->val->dim; i++) {
    if (var->children[0]->val->shape[i] != temp->shape[i]) {
      ndarray *temp2 = sum_ndarray(reduced_grad, i);
      free_ndarray(&reduced_grad);
      reduced_grad = temp2;
    }
  }
  var->children[0]->grad =
      add_ndarray_ndarray(reduced_grad, var->children[0]->grad);
  free_ndarray(&temp);
  free_ndarray(&place_holder);
  free_ndarray(&reduced_grad);

  place_holder = var->children[1]->grad;
  temp = multiply_ndarray_ndarray(var->children[0]->val, var->grad);
  reduced_grad = copy_ndarray(temp);
  for (int i = 0; i < var->children[1]->val->dim; i++) {
    if (var->children[1]->val->shape[i] != temp->shape[i]) {
      ndarray *temp2 = sum_ndarray(reduced_grad, i);
      free_ndarray(&reduced_grad);
      reduced_grad = temp2;
    }
  }
  var->children[1]->grad =
      add_ndarray_ndarray(reduced_grad, var->children[1]->grad);
  free_ndarray(&temp);
  free_ndarray(&place_holder);
  free_ndarray(&reduced_grad);
}

void divide_backward(variable *var) {
  ndarray *place_holder;
  ndarray *temp0;
  ndarray *temp1;
  ndarray *reduced_grad;

  place_holder = var->children[0]->grad;
  temp0 = divide_scalar_ndarray(var->children[1]->val, 1.0);
  temp1 = multiply_ndarray_ndarray(temp0, var->grad);
  reduced_grad = copy_ndarray(temp1);
  for (int i = 0; i < var->children[0]->val->dim; i++) {
    if (var->children[0]->val->shape[i] != temp1->shape[i]) {
      ndarray *temp2 = sum_ndarray(reduced_grad, i);
      free_ndarray(&reduced_grad);
      reduced_grad = temp2;
    }
  }
  var->children[0]->grad =
      add_ndarray_ndarray(reduced_grad, var->children[0]->grad);
  free_ndarray(&temp0);
  free_ndarray(&temp1);
  free_ndarray(&place_holder);
  free_ndarray(&reduced_grad);

  place_holder = var->children[1]->grad;
  temp0 =
      multiply_ndarray_ndarray(var->children[1]->val, var->children[1]->val);
  temp1 = divide_ndarray_ndarray(var->children[0]->val, temp0);
  free_ndarray(&temp0);
  temp0 = multiply_ndarray_scalar(temp1, -1.0);
  free_ndarray(&temp1);
  temp1 = multiply_ndarray_ndarray(temp0, var->grad);
  reduced_grad = copy_ndarray(temp1);
  for (int i = 0; i < var->children[1]->val->dim; i++) {
    if (var->children[1]->val->shape[i] != temp1->shape[i]) {
      ndarray *temp2 = sum_ndarray(reduced_grad, i);
      free_ndarray(&reduced_grad);
      reduced_grad = temp2;
    }
  }
  var->children[1]->grad =
      add_ndarray_ndarray(reduced_grad, var->children[1]->grad);
  free_ndarray(&temp0);
  free_ndarray(&temp1);
  free_ndarray(&place_holder);
  free_ndarray(&reduced_grad);
}

void power_backward(variable *var) {
  ndarray *place_holder;
  ndarray *temp0;
  ndarray *temp1;
  ndarray *temp2;
  ndarray *reduced_grad;

  place_holder = var->children[0]->grad;
  temp0 = subtract_ndarray_scalar(var->children[1]->val, 1);
  temp1 = power_ndarray_ndarray(var->children[0]->val, temp0);
  free_ndarray(&temp0);
  temp0 = multiply_ndarray_ndarray(var->children[1]->val, temp1);
  free_ndarray(&temp1);
  temp1 = multiply_ndarray_ndarray(temp0, var->grad);
  reduced_grad = copy_ndarray(temp1);
  for (int i = 0; i < var->children[0]->val->dim; i++) {
    if (var->children[0]->val->shape[i] != temp1->shape[i]) {
      ndarray *temp2 = sum_ndarray(reduced_grad, i);
      free_ndarray(&reduced_grad);
      reduced_grad = temp2;
    }
  }
  var->children[0]->grad =
      add_ndarray_ndarray(reduced_grad, var->children[0]->grad);
  free_ndarray(&temp0);
  free_ndarray(&temp1);
  free_ndarray(&place_holder);
  free_ndarray(&reduced_grad);

  place_holder = var->children[1]->grad;
  temp0 = log_ndarray(var->children[0]->val);
  temp1 = multiply_ndarray_ndarray(temp0, var->grad);
  free_ndarray(&temp0);
  temp2 = power_ndarray_ndarray(var->children[0]->val, var->children[1]->val);
  temp0 = multiply_ndarray_ndarray(temp2, temp1);
  reduced_grad = copy_ndarray(temp0);
  for (int i = 0; i < var->children[1]->val->dim; i++) {
    if (var->children[1]->val->shape[i] != temp0->shape[i]) {
      ndarray *temp3 = sum_ndarray(reduced_grad, i);
      free_ndarray(&reduced_grad);
      reduced_grad = temp3;
    }
  }
  var->children[1]->grad =
      add_ndarray_ndarray(reduced_grad, var->children[1]->grad);
  free_ndarray(&temp0);
  free_ndarray(&temp1);
  free_ndarray(&temp2);
  free_ndarray(&place_holder);
  free_ndarray(&reduced_grad);
}

void exp_backward(variable *var) {
  ndarray *place_holder;
  ndarray *temp0;
  ndarray *temp1;

  place_holder = var->children[0]->grad;
  temp0 = unary_op_ndarray(var->children[0]->val, exp);
  temp1 = multiply_ndarray_ndarray(var->grad, temp0);
  var->children[0]->grad = add_ndarray_ndarray(var->children[0]->grad, temp1);
  free_ndarray(&temp0);
  free_ndarray(&temp1);
  free_ndarray(&place_holder);
}

void sum_backward(variable *var) {
  ndarray *place_holder;

  place_holder = var->children[0]->grad;
  var->children[0]->grad =
      add_ndarray_ndarray(var->children[0]->grad, var->grad);

  free_ndarray(&place_holder);
}

void relu_backward(variable *var) {
  ndarray *place_holder;
  ndarray *temp0;
  ndarray *temp1;

  place_holder = var->children[0]->grad;
  temp0 = empty_like_ndarray(var->children[0]->val);
  for (int i = 0; i < temp0->size; i++) {
    if (var->children[0]->val->data[i] > 0.0) {
      temp0->data[i] = 1.0;
    } else {
      temp0->data[i] = 0.0;
    }
  }
  temp1 = multiply_ndarray_ndarray(var->grad, temp0);
  var->children[0]->grad = add_ndarray_ndarray(var->children[0]->grad, temp1);
  free_ndarray(&temp0);
  free_ndarray(&temp1);
  free_ndarray(&place_holder);
}

void sigmoid_backward(variable *var) {
  ndarray *place_holder;
  ndarray *temp0;
  ndarray *temp1;

  place_holder = var->children[0]->grad;
  temp0 = unary_op_ndarray(var->children[0]->val, d_sigmoid);
  temp1 = multiply_ndarray_ndarray(var->grad, temp0);
  var->children[0]->grad = add_ndarray_ndarray(var->children[0]->grad, temp1);
  free_ndarray(&temp0);
  free_ndarray(&temp1);
  free_ndarray(&place_holder);
}

void tanh_backward(variable *var) {
  ndarray *place_holder;
  ndarray *temp0;
  ndarray *temp1;

  place_holder = var->children[0]->grad;
  temp0 = unary_op_ndarray(var->children[0]->val, d_tanh);
  temp1 = multiply_ndarray_ndarray(var->grad, temp0);
  var->children[0]->grad = add_ndarray_ndarray(var->children[0]->grad, temp1);
  free_ndarray(&temp0);
  free_ndarray(&temp1);
  free_ndarray(&place_holder);
}

void matmul_backward(variable *var) {
  int dim;
  int *order;
  ndarray *place_holder;
  ndarray *temp0;
  ndarray *temp1;

  place_holder = var->children[0]->grad;
  dim = var->children[1]->val->dim;
  order = (int *)malloc(dim * sizeof(int));
  for (int i = 0; i < dim - 2; i++) {
    order[i] = i;
  }
  order[dim - 2] = dim - 1;
  order[dim - 1] = dim - 2;
  temp0 = transpose_ndarray(var->children[1]->val, order);
  free(order);
  temp1 = matmul_ndarray(var->grad, temp0);
  var->children[0]->grad = add_ndarray_ndarray(temp1, var->children[0]->grad);
  free_ndarray(&temp0);
  free_ndarray(&temp1);
  free_ndarray(&place_holder);

  place_holder = var->children[1]->grad;
  dim = var->children[0]->val->dim;
  order = (int *)malloc(dim * sizeof(int));
  for (int i = 0; i < dim - 2; i++) {
    order[i] = i;
  }
  order[dim - 2] = dim - 1;
  order[dim - 1] = dim - 2;
  temp0 = transpose_ndarray(var->children[0]->val, order);
  free(order);
  temp1 = matmul_ndarray(temp0, var->grad);
  var->children[1]->grad = add_ndarray_ndarray(temp1, var->children[1]->grad);
  free_ndarray(&temp0);
  free_ndarray(&temp1);
  free_ndarray(&place_holder);
}

variable *add_variable(variable *var1, variable *var2) {
  variable *var = (variable *)malloc(sizeof(variable));
  var->val = add_ndarray_ndarray(var1->val, var2->val);
  var->grad = zeros_ndarray(var->val->dim, var->val->shape);
  var->children = (variable **)malloc(2 * sizeof(variable *));
  var->children[0] = var1;
  var->children[1] = var2;
  var->n_children = 2;
  var->backward = add_backward;
  var->ref_count = 0;
  var1->ref_count++;
  var2->ref_count++;

  return var;
}

variable *subtract_variable(variable *var1, variable *var2) {
  variable *var = (variable *)malloc(sizeof(variable));
  var->val = subtract_ndarray_ndarray(var1->val, var2->val);
  var->grad = zeros_ndarray(var->val->dim, var->val->shape);
  var->children = (variable **)malloc(2 * sizeof(variable *));
  var->children[0] = var1;
  var->children[1] = var2;
  var->n_children = 2;
  var->backward = subtract_backward;
  var->ref_count = 0;
  var1->ref_count++;
  var2->ref_count++;

  return var;
}

variable *multiply_variable(variable *var1, variable *var2) {
  variable *var = (variable *)malloc(sizeof(variable));
  var->val = multiply_ndarray_ndarray(var1->val, var2->val);
  var->grad = zeros_ndarray(var->val->dim, var->val->shape);
  var->children = (variable **)malloc(2 * sizeof(variable *));
  var->children[0] = var1;
  var->children[1] = var2;
  var->n_children = 2;
  var->backward = multiply_backward;
  var->ref_count = 0;
  var1->ref_count++;
  var2->ref_count++;

  return var;
}

variable *divide_variable(variable *var1, variable *var2) {
  variable *var = (variable *)malloc(sizeof(variable));
  var->val = divide_ndarray_ndarray(var1->val, var2->val);
  var->grad = zeros_ndarray(var->val->dim, var->val->shape);
  var->children = (variable **)malloc(2 * sizeof(variable *));
  var->children[0] = var1;
  var->children[1] = var2;
  var->n_children = 2;
  var->backward = divide_backward;
  var->ref_count = 0;
  var1->ref_count++;
  var2->ref_count++;

  return var;
}

variable *power_variable(variable *var1, variable *var2) {
  variable *var = (variable *)malloc(sizeof(variable));
  var->val = power_ndarray_ndarray(var1->val, var2->val);
  var->grad = zeros_ndarray(var->val->dim, var->val->shape);
  var->children = (variable **)malloc(2 * sizeof(variable *));
  var->children[0] = var1;
  var->children[1] = var2;
  var->n_children = 2;
  var->backward = power_backward;
  var->ref_count = 0;
  var1->ref_count++;
  var2->ref_count++;

  return var;
}

variable *exp_variable(variable *var) {
  variable *n_var = (variable *)malloc(sizeof(variable));
  n_var->val = unary_op_ndarray(var->val, exp);
  n_var->grad = zeros_ndarray(n_var->val->dim, n_var->val->shape);
  n_var->children = (variable **)malloc(sizeof(variable *));
  n_var->children[0] = var;
  n_var->n_children = 1;
  n_var->backward = exp_backward;
  n_var->ref_count = 0;
  var->ref_count++;

  return n_var;
}

variable *sum_variable(variable *var, int axis) {
  variable *n_var = (variable *)malloc(sizeof(variable));
  n_var->val = sum_ndarray(var->val, axis);
  n_var->grad = zeros_ndarray(n_var->val->dim, n_var->val->shape);
  n_var->children = (variable **)malloc(sizeof(variable *));
  n_var->children[0] = var;
  n_var->n_children = 1;
  n_var->backward = sum_backward;
  n_var->ref_count = 0;
  var->ref_count++;

  return n_var;
}

variable *relu_variable(variable *var) {
  variable *n_var = (variable *)malloc(sizeof(variable));
  n_var->val = unary_op_ndarray(var->val, relu);
  n_var->grad = zeros_ndarray(n_var->val->dim, n_var->val->shape);
  n_var->children = (variable **)malloc(sizeof(variable *));
  n_var->children[0] = var;
  n_var->n_children = 1;
  n_var->backward = relu_backward;
  n_var->ref_count = 0;
  var->ref_count++;

  return n_var;
}

variable *sigmoid_variable(variable *var) {
  variable *n_var = (variable *)malloc(sizeof(variable));
  n_var->val = unary_op_ndarray(var->val, sigmoid);
  n_var->grad = zeros_ndarray(n_var->val->dim, n_var->val->shape);
  n_var->children = (variable **)malloc(sizeof(variable *));
  n_var->children[0] = var;
  n_var->n_children = 1;
  n_var->backward = sigmoid_backward;
  n_var->ref_count = 0;
  var->ref_count++;

  return n_var;
}

variable *softmax_variable(variable *var, int axis) {
  variable *exp_var = exp_variable(var);
  variable *sum_exp_var = sum_variable(exp_var, axis);
  return divide_variable(exp_var, sum_exp_var);
}

variable *tanh_variable(variable *var) {
  variable *n_var = (variable *)malloc(sizeof(variable));
  n_var->val = unary_op_ndarray(var->val, tanh);
  n_var->grad = zeros_ndarray(n_var->val->dim, n_var->val->shape);
  n_var->children = (variable **)malloc(sizeof(variable *));
  n_var->children[0] = var;
  n_var->n_children = 1;
  n_var->backward = tanh_backward;
  n_var->ref_count = 0;
  var->ref_count++;

  return n_var;
}

variable *matmul_variable(variable *var1, variable *var2) {
  variable *var = (variable *)malloc(sizeof(variable));
  var->val = matmul_ndarray(var1->val, var2->val);
  var->grad = zeros_ndarray(var->val->dim, var->val->shape);
  var->children = (variable **)malloc(2 * sizeof(variable *));
  var->children[0] = var1;
  var->children[1] = var2;
  var->n_children = 2;
  var->backward = matmul_backward;
  var->ref_count = 0;
  var1->ref_count++;
  var2->ref_count++;

  return var;
}

void build_topology(variable *var, variable ***topology, int *topology_size,
                    variable ***visited, int *visited_size) {
  for (int i = 0; i < *visited_size; ++i) {
    if ((*visited)[i] == var) {
      return;
    }
  }

  *visited =
      (variable **)realloc(*visited, (*visited_size + 1) * sizeof(variable *));
  (*visited)[*visited_size] = var;
  (*visited_size)++;

  for (int i = 0; i < var->n_children; ++i) {
    build_topology(var->children[i], topology, topology_size, visited,
                   visited_size);
  }

  *topology = (variable **)realloc(*topology,
                                   (*topology_size + 1) * sizeof(variable *));
  (*topology)[*topology_size] = var;
  (*topology_size)++;
}

void backward_variable(variable *root_var) {
  variable **topology = NULL;
  int topology_size = 0;
  variable **visited = NULL;
  int visited_size = 0;

  build_topology(root_var, &topology, &topology_size, &visited, &visited_size);

  for (int i = topology_size - 1; i >= 0; --i) {
    if (topology[i]->backward) {
      topology[i]->backward(topology[i]);
    }
  }
  free(topology);
  free(visited);
}

void print_variable(variable *var) {
  printf("----data----\n");
  print_ndarray(var->val);
  print_ndarray(var->grad);
  printf("------------\n");
  printf("Number of children: %d\n", var->n_children);
  for (int i = 0; i < var->n_children; i++) {
    printf("Variable %d:\n", i + 1);
    print_variable(var->children[i]);
  }
}

void free_variable(variable **var) {
  if (*var == NULL) {
    return;
  }

  (*var)->ref_count--;
  if ((*var)->ref_count <= 0) {
    if ((*var)->val != NULL) {
      free_ndarray(&((*var)->val));
    }

    if ((*var)->grad != NULL) {
      free_ndarray(&((*var)->grad));
    }

    if ((*var)->children != NULL) {
      free((*var)->children);
    }

    free(*var);
    *var = NULL;
  }
}

void free_graph_variable(variable **root_var) {
  if (*root_var == NULL) {
    return;
  }

  for (int i = 0; i < (*root_var)->n_children; i++) {
    free_graph_variable(&((*root_var)->children[i]));
  }

  free_variable(root_var);
}
