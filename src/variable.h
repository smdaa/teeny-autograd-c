#include "ndarray.h"

#ifndef TEENY_AUTOGRAD_C_VARIABLE_H
#define TEENY_AUTOGRAD_C_VARIABLE_H

typedef enum {
  OK_TO_FREE,
  DO_NOT_FREE
} variable_tag;

typedef struct variable {
  ndarray *val;
  ndarray *grad;
  struct variable **children;
  int n_children;

  void (*backward)(struct variable *);

  int ref_count;
  variable_tag tag;
} variable;

variable *new_variable(ndarray *val);

variable *add_variable(variable *var1, variable *var2);

variable *subtract_variable(variable *var1, variable *var2);

variable *multiply_variable(variable *var1, variable *var2);

variable *divide_variable(variable *var1, variable *var2);

variable *power_variable(variable *var1, variable *var2);

variable *negate_variable(variable *var);

variable *exp_variable(variable *var);

variable *log_variable(variable *var);

variable *sum_variable(variable *var, int axis);

variable *relu_variable(variable *var);

variable *sigmoid_variable(variable *var);

variable *softmax_variable(variable *var, int axis);

variable *tanh_variable(variable *var);

variable *matmul_variable(variable *var1, variable *var2);

void backward_variable(variable *root_var);

void print_variable(variable *var);

void free_variable(variable **var);

void free_graph_variable(variable **root_var);

#endif
