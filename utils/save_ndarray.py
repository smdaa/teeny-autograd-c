import numpy as np


def save_ndarray_to_file(filename, arr):
    with open(filename, "w") as file:
        dim = arr.ndim
        size = arr.size
        shape = " ".join(map(str, arr.shape))
        data = "\n".join(map(str, arr.flatten()))

        file.write(f"{dim} {size}\n")
        file.write(f"{shape}\n")
        file.write(f"{data}\n")
