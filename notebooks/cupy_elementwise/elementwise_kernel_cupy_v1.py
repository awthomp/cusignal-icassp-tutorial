import cupy as cp
import numpy as np
import sys

from cupy import prof
from scipy import signal

# Elementwise kernel implementation of CuPy

_gauss_spline_kernel = cp.ElementwiseKernel(
    "T x, int64 n",
    "T output",
    """
    output = 1 / sqrt( 2.0 * M_PI * signsq ) \
        * exp( -( x * x ) * r_signsq );
    """,
    "_gauss_spline_kernel",
    options=("-std=c++11",),
    loop_prep="const double signsq { ( n + 1 ) / 12.0 }; \
               const double r_signsq { 0.5 / signsq };",
)


def gauss_spline(x, n):
    return _gauss_spline_kernel(x, n)


def rand_data_gen_gpu(num_samps, dim=1, dtype=np.float64):
    inp = tuple(np.ones(dim, dtype=int) * num_samps)
    cpu_sig = np.random.random(inp)
    cpu_sig = cpu_sig.astype(dtype)
    gpu_sig = cp.asarray(cpu_sig)

    return cpu_sig, gpu_sig


def main():
    loops = int(sys.argv[1])

    n = np.random.randint(0, 1234)

    num_samps = 2 ** 16
    cpu_sig, gpu_sig = rand_data_gen_gpu(num_samps)

    # Run baseline with scipy.signal.gauss_spline
    with prof.time_range("scipy_gauss_spline", 0):
        cpu_gauss_spline = signal.gauss_spline(cpu_sig, n)

    # Run CuPy version
    with prof.time_range("cupy_gauss_spline", 1):
        gpu_gauss_spline = gauss_spline(gpu_sig, n)

    # Compare results
    np.testing.assert_allclose(
        cpu_gauss_spline, cp.asnumpy(gpu_gauss_spline), 1e-3
    )

    # Run multiple passes to get average
    for _ in range(loops):
        with prof.time_range("cupy_gauss_spline_loop", 2):
            gpu_gauss_spline = gauss_spline(gpu_sig, n)
            cp.cuda.runtime.deviceSynchronize()

if __name__ == "__main__":
    sys.exit(main())