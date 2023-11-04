CC=gcc
CFLAGS=-std=c11 -Wall -Wextra -g
LDFLAGS=-lcmocka -lm

all: test_ndarray test_variable test_linear_layer

test_ndarray: test_ndarray.c ../src/ndarray.c
	$(CC) $(CFLAGS) -o test_ndarray test_ndarray.c ../src/ndarray.c $(LDFLAGS)

test_variable: test_variable.c ../src/ndarray.c ../src/variable.c
	$(CC) $(CFLAGS) -o test_variable test_variable.c ../src/ndarray.c ../src/variable.c $(LDFLAGS)

test_linear_layer: test_linear_layer.c ../src/ndarray.c ../src/variable.c ../src/linear_layer.c
	$(CC) $(CFLAGS) -o test_linear_layer test_linear_layer.c  ../src/ndarray.c ../src/variable.c ../src/linear_layer.c $(LDFLAGS)

.PHONY: clean

clean:
	rm -f test_ndarray test_variable test_linear_layer
