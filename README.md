<p align="center">
    <h1 align="center">TEENY-AUTOGRAD-C</h1>
</p>
<p align="center">
    <em><code>► Small Autograd Library from Scratch in C</code></em>
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#-overview)
- [ Repository Structure](#-repository-structure)
- [ Getting Started](#-getting-started)
  - [ Setup](#-installation)
  - [ Tests](#-tests)
- [ License](#-license)
</details>
<hr>

##  Overview

Autograd, is a fundamental component in machine learning frameworks, enabling the automatic computation of gradients for training neural networks. This repository is my attempt at writing an Autograd library from scratch (no third-party libraries) in pure C.

---


##  Repository Structure

```sh
└── teeny-autograd-c/
    ├── .github
    │   └── workflows
    ├── CMakeLists.txt
    ├── examples
    │   ├── mnist_mlp
    │   └── paint
    ├── extern
    ├── src
    │   ├── multilayer_perceptron.c
    │   ├── multilayer_perceptron.h
    │   ├── ndarray.c
    │   ├── ndarray.h
    │   ├── variable.c
    │   └── variable.h
    └── test
        ├── generate_multilayer_perceptron_tests.py
        ├── generate_ndarray_tests.py
        ├── generate_variable_tests.py
        ├── save_ndarray.py
        ├── test_data
        ├── test_multilayer_perceptron.c
        ├── test_ndarray.c
        └── test_variable.c
```

---

##  Getting Started

###  Setup

> 1. Clone the teeny-autograd-c repository:
>
> ```console
> $ git clone https://github.com/smdaa/teeny-autograd-c
> $ cd teeny-autograd-c
> $ git submodule init && git submodule update --recursive
> $ git lfs fetch --all
> $ git lfs pull
> ```
>
> 3. Make and cd to build directory:
> ```console
> $ mkdir build
> $ cd build
> ```
>
> 4. Build project:
> ```console
> $ cmake -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release ..
> $ make

###  Tests

> Run the test suite using the command below:
> ```console
> $  ctest --rerun-failed --output-on-failure
> ```

---

##  License

This project is protected under the [The Unlicense](https://choosealicense.com/licenses/unlicense/) License.

---


