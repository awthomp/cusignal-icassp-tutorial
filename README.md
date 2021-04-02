# ICASSP '21 Tutorial: GPU-Acceleration of Signal Processing Workflows from Python

## Welcome
Welcome to our tutorial on GPU-Accelerated Signal Processing with cuSignal! [cuSignal](https://github.com/rapidsai/cusignal) is a free and open-source software library designed to support and extend [SciPy Signal](https://docs.scipy.org/doc/scipy/reference/signal.html) functionality towards [NVIDIA](https://www.nvidia.com/en-us/) GPUs. Housed under NVIDIA's [RAPIDS](https://rapids.ai/) open data science project, cuSignal delivers 100-300x speedups over CPU with a fully Pythonic API. In this tutorial, we will:
- Introduce the audience to the cuSignal design philosophy, key features, and community success stories
- Demonstrate how we built cuSignal - including leveraging existing GPU-accelerated libraries like [CuPy](https://cupy.dev/) and how to build your own custom CUDA functions with [Numba](https://numba.pydata.org/), CuPy Elementwise Kernels, and -- for maximum speed -- CuPy CUDA Raw Modules
- Walk through an end-to-end example including cuSignal based pre-processing and AI training/inferencing with a simple PyTorch-based neural network

Our goal is to provide an interactive and collaborative tutorial, full of GPU-goodies, best practices, and showing that you really can achieve eye-popping speedups with Python. We want to show the ease and flexibility of creating and implementing GPU-based high performance signal processing workloads from Python, and you can expect to learn as much about *using* cuSignal as to *extending* cuSignal via your own Python-CUDA code.

We know that 2020 has been *a year* to say the least, and it goes without saying that we wish we could all be in Toronto together. We hope that everyone continues to remain safe and healthy.

Let's get started.

## Presentation Materials
Before we jump into code, we'll be walking through a presentation concerning the usecases, features, performance, and techncial backend of cuSignal. You can find a copy of these slides [here](UPDATE FUTURE URL).

## Installing cuSignal
cuSignal has been tested on and supports all modern GPUs - from Maxwell to Ampere. While Anaconda is the preferred installation mechanism for cuSignal, developers and Jetson users should follow the source build instructions below. As of cuSignal 0.16, there isn't a cuSignal conda package for aarch64. In general, it's assumed that the developer has already installed the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and associated GPU drivers.

Complete build instructions can be found on [cuSignal's installation README](https://github.com/rapidsai/cusignal#installation), but we will highlight Anaconda Linux builds and No GPU instuctions below.

* [Conda: Linux OS](#conda-linux-os)
* [No GPU](#no-gpu)

### Conda, Linux OS
cuSignal can be installed with conda ([Miniconda](https://docs.conda.io/en/latest/miniconda.html), or the full [Anaconda distribution](https://www.anaconda.com/distribution/)) from the `rapidsai` channel. Once installed, create a new cusignal environment and install the package. Instructions for doing this are shown below.

```
# Create Conda Environment
conda create --name cusignal python=3.8

# Activate Conda Environment
conda activate cusignal

# Install cuSignal into cusignal Environment
conda install -c rapidsai cusignal

# Confirm cuSignal and its Dependencies Successfully Installed
python
>>> import cusignal
>>> import cupy as cp
>>> from numba import cuda
```

### No GPU
No GPU? No problem. We can use Google Colab for access to a no-cost GPU instance.
1. Navigate to Colab via [this link](https://colab.research.google.com/notebooks/intro.ipynb). If you're unfamiliar with notebook based programming (or Colab in general), feel free to take a few minutes to explore the getting started guide.
2. Click `File -> New Notebook` to create a new Colab notebook.
3. Install cuSignal

## Notebooks Used in Today's Tutorial
* [Introduction to cuSignal]
* [CuPy Overview and Elementwise Kernels]
* [Numba Overview and Custom CUDA Kernels]
* [Brief Introduction of Raw CUDA Kernels]
* [Putting It All Together]
