import numpy as np
import sys
from cupy import prof
from scipy import signal

# Naive serial implementation of Python

def rand_data_gen(num_samps, dim=1, dtype=np.float64):
    inp = tuple(np.ones(dim, dtype=int) * num_samps)
    cpu_sig = np.random.random(inp)
    cpu_sig = cpu_sig.astype(dtype)

    return cpu_sig


def main():

    loops = int(sys.argv[1])

    n = np.random.randint(0, 1234)

    num_samps = 2 ** 16
    cpu_sig = rand_data_gen(num_samps)

    # Run baseline with scipy.signal.gauss_spline
    with prof.time_range("scipy_gauss_spline", 0):
        cpu_gauss_spline = signal.gauss_spline(cpu_sig, n)

    # Run multiple passes to get average
    for _ in range(loops):
        with prof.time_range("scipy_gauss_spline_loop", 0):
            cpu_gauss_spline = signal.gauss_spline(cpu_sig, n)

if __name__ == "__main__":
    sys.exit(main())