CC=gcc
CFLAGS=-std=c11 -Wall -Wextra -O3 
LDFLAGS=-lm -lpng
SRC=../../src
OBJ=$(SRC)/ndarray.o $(SRC)/variable.o $(SRC)/multilayer_perceptron.o

all: main

$(SRC)/%.o: $(SRC)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

main: main.c $(OBJ)
	$(CC) $(CFLAGS) -o main main.c $(OBJ) $(LDFLAGS)

.PHONY: clean

clean:
	-rm -f main $(OBJ)