import cupy as cp
import numpy as np
import sys

from cupy import prof

# Elementwise kernel for multiple CuPy calls

_signal_kernel = cp.ElementwiseKernel(
    "T signal",
    "float64 amp, float64 phase, float64 real, float64 imag",
    """
    amp = sqrt((signal * conj(signal)).real());
    phase = arg(signal);
    real = signal.real();
    imag = signal.imag();
    """,
    "_signal_kernel",
    options=("-std=c++11",),
)


def signal(sig):
    return _signal_kernel(sig)


def cupy_signal(signal):
    amp = cp.sqrt(cp.real(signal * cp.conj(signal)))
    phase = cp.angle(signal)
    real = cp.real(signal)
    imag = cp.imag(signal)

    return amp, phase, real, imag


def main():
    loops = int(sys.argv[1])

    num_samps = 2 ** 16

    cpu_sig = np.random.rand(num_samps) + 1.0j * np.random.rand(num_samps)
    gpu_sig = cp.array(cpu_sig)

    # Run baseline with cupy_signal
    with prof.time_range("CuPy signal", 0):
        amp, phase, real, imag = cupy_signal(gpu_sig)

    # Run EWK version
    with prof.time_range("EWK signal", 1):
        amp_EWK, phase_EWK, real_EWK, imag_EWK = signal(gpu_sig)

    # Compare results
    cp.testing.assert_allclose(amp, amp_EWK, 1e-3)
    cp.testing.assert_allclose(phase, phase_EWK, 1e-3)
    cp.testing.assert_allclose(real, real_EWK, 1e-3)
    cp.testing.assert_allclose(imag, imag_EWK, 1e-3)

    # Run multiple passes to get average
    for _ in range(loops):
        with prof.time_range("cupy_signal_avg", 2):
            amp, phase, real, imag = cupy_signal(gpu_sig)
            cp.cuda.runtime.deviceSynchronize()


    # Run multiple passes to get average
    for _ in range(loops):
        with prof.time_range("ewk_signal_avg", 3):
            amp_EWK, phase_EWK, real_EWK, imag_EWK = signal(gpu_sig)
            cp.cuda.runtime.deviceSynchronize()

if __name__ == "__main__":
    sys.exit(main())