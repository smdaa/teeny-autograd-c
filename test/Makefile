CC=gcc
CFLAGS=-std=c11 -Wall -Wextra -g
LDFLAGS=-lcmocka -lm

all: test_ndarray test_variable test_multilayer_perceptron

test_ndarray: test_ndarray.c ../src/ndarray.c
	$(CC) $(CFLAGS) -o test_ndarray test_ndarray.c ../src/ndarray.c $(LDFLAGS)

test_variable: test_variable.c ../src/ndarray.c ../src/variable.c
	$(CC) $(CFLAGS) -o test_variable test_variable.c ../src/ndarray.c ../src/variable.c $(LDFLAGS)

test_multilayer_perceptron: test_multilayer_perceptron.c ../src/ndarray.c ../src/variable.c ../src/multilayer_perceptron.c
	$(CC) $(CFLAGS) -o test_multilayer_perceptron test_multilayer_perceptron.c  ../src/ndarray.c ../src/variable.c ../src/multilayer_perceptron.c $(LDFLAGS)

.PHONY: clean

clean:
	rm -f test_ndarray test_variable test_multilayer_perceptron
