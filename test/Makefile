CC=gcc
CFLAGS=-std=c11 -Wall -Wextra -g
LDFLAGS=-lcmocka -lm
SRC=../src
OBJ=$(SRC)/ndarray.o $(SRC)/variable.o $(SRC)/multilayer_perceptron.o

all: test_ndarray test_variable test_multilayer_perceptron

$(SRC)/%.o: $(SRC)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

test_ndarray: test_ndarray.c $(SRC)/ndarray.o
	$(CC) $(CFLAGS) -o test_ndarray test_ndarray.c $(SRC)/ndarray.o $(LDFLAGS)

test_variable: test_variable.c $(OBJ)
	$(CC) $(CFLAGS) -o test_variable test_variable.c $(OBJ) $(LDFLAGS)

test_multilayer_perceptron: test_multilayer_perceptron.c $(OBJ)
	$(CC) $(CFLAGS) -o test_multilayer_perceptron test_multilayer_perceptron.c $(OBJ) $(LDFLAGS)

.PHONY: clean

clean:
	-rm -f test_ndarray test_variable test_multilayer_perceptron $(OBJ)