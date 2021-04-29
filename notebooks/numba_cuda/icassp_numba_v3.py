# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import numpy as np
import sys

from cupy import prof
from math import sin, cos, atan2
from numba import cuda, void, int32, float32, float64, boolean
from scipy import signal


# Numba: Version 3
# Implementations a user level cache from version 2
# and seperates 32 bit and 64 bit versions to 
# reduce register pressure.


_kernel_cache = {}


def _numba_lombscargle_32(x, y, freqs, pgram, y_dot):

    dtype = float32

    F = int32(cuda.grid(1))
    strideF = int32(cuda.gridsize(1))

    if not y_dot[0]:
        yD = dtype(1.0)
    else:
        yD = dtype(2.0 / y_dot[0])

    for i in range(F, int32(freqs.shape[0]), strideF):

        # Copy data to registers
        freq = dtype(freqs[i])

        xc = dtype(0.0)
        xs = dtype(0.0)
        cc = dtype(0.0)
        ss = dtype(0.0)
        cs = dtype(0.0)

        for j in range(int32(x.shape[0])):

            c = dtype(cos(dtype(freq * x[j])))
            s = dtype(sin(dtype(freq * x[j])))

            xc += dtype(y[j] * c)
            xs += dtype(y[j] * s)
            cc += dtype(c * c)
            ss += dtype(s * s)
            cs += dtype(c * s)

        tau = dtype(atan2(dtype(2.0 * cs), dtype(cc - ss)) / dtype(2.0 * freq))
        c_tau = dtype(cos(freq * tau))
        s_tau = dtype(sin(freq * tau))
        c_tau2 = dtype(c_tau * c_tau)
        s_tau2 = dtype(s_tau * s_tau)
        cs_tau = dtype(2.0 * dtype(c_tau * s_tau))

        pgram[i] = dtype(
            0.5
            * dtype(
                dtype(
                    dtype(dtype(c_tau * xc) + dtype(s_tau * xs)) ** 2
                    / dtype(dtype(c_tau2 * cc) + dtype(cs_tau * cs) + dtype(s_tau2 * ss))
                )
                + dtype(
                    dtype(dtype(c_tau * xs) - dtype(s_tau * xc)) ** 2
                    / dtype(dtype(c_tau2 * ss) - dtype(cs_tau * cs) + dtype(s_tau2 * cc))
                )
            )
        ) * yD


def _numba_lombscargle_64(x, y, freqs, pgram, y_dot):

    dtype = float64

    F = int32(cuda.grid(1))
    strideF = int32(cuda.gridsize(1))

    if not y_dot[0]:
        yD = dtype(1.0)
    else:
        yD = dtype(2.0 / y_dot[0])

    for i in range(F, int32(freqs.shape[0]), strideF):

        # Copy data to registers
        freq = dtype(freqs[i])

        xc = dtype(0.0)
        xs = dtype(0.0)
        cc = dtype(0.0)
        ss = dtype(0.0)
        cs = dtype(0.0)

        for j in range(int32(x.shape[0])):

            c = dtype(cos(dtype(freq * x[j])))
            s = dtype(sin(dtype(freq * x[j])))

            xc += dtype(y[j] * c)
            xs += dtype(y[j] * s)
            cc += dtype(c * c)
            ss += dtype(s * s)
            cs += dtype(c * s)

        tau = dtype(atan2(dtype(2.0 * cs), dtype(cc - ss)) / dtype(2.0 * freq))
        c_tau = dtype(cos(freq * tau))
        s_tau = dtype(sin(freq * tau))
        c_tau2 = dtype(c_tau * c_tau)
        s_tau2 = dtype(s_tau * s_tau)
        cs_tau = dtype(2.0 * dtype(c_tau * s_tau))

        pgram[i] = dtype(
            0.5
            * dtype(
                dtype(
                    dtype(dtype(c_tau * xc) + dtype(s_tau * xs)) ** 2
                    / dtype(dtype(c_tau2 * cc) + dtype(cs_tau * cs) + dtype(s_tau2 * ss))
                )
                + dtype(
                    dtype(dtype(c_tau * xs) - dtype(s_tau * xc)) ** 2
                    / dtype(dtype(c_tau2 * ss) - dtype(cs_tau * cs) + dtype(s_tau2 * cc))
                )
            )
        ) * yD


def _numba_lombscargle_signature(ty):
    return void(
        ty[::1], ty[::1], ty[::1], ty[::1], ty[::1],  # x  # y  # freqs  # pgram  # y_dot
    )


def _lombscargle(x, y, freqs, pgram, y_dot):

    if (pgram.dtype == 'float32'):
        numba_type = float32
    elif (pgram.dtype == 'float64'):
        numba_type = float64

    if (str(numba_type)) in _kernel_cache:
        kernel = _kernel_cache[(str(numba_type))]
    else:
        sig = _numba_lombscargle_signature(numba_type)
        if (pgram.dtype == 'float32'):
            kernel = _kernel_cache[(str(numba_type))] = cuda.jit(sig)(_numba_lombscargle_32)
            print("Registers", kernel._func.get().attrs.regs)
        elif (pgram.dtype == 'float64'):
            kernel = _kernel_cache[(str(numba_type))] = cuda.jit(sig)(_numba_lombscargle_64)
            print("Registers", kernel._func.get().attrs.regs)

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    threadsperblock = (128, )
    blockspergrid = (numSM * 20,)

    kernel[blockspergrid, threadsperblock](x, y, freqs, pgram, y_dot)

    cuda.synchronize()


def lombscargle(
    x,
    y,
    freqs,
    precenter=False,
    normalize=False,
):

    pgram = cuda.device_array_like(freqs)

    assert x.ndim == 1
    assert y.ndim == 1
    assert freqs.ndim == 1

    # Check input sizes
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")

    y_dot = cuda.device_array(shape=(1,), dtype=y.dtype)
    if normalize:
        cp.dot(y, y, out=y_dot)

    if precenter:
        y_in = y - y.mean()
    else:
        y_in = y

    _lombscargle(x, y_in, freqs, pgram, y_dot)

    return pgram


if __name__ == "__main__":

    dtype = sys.argv[1]
    loops = int(sys.argv[2])

    A = 2.0
    w = 1.0
    phi = 0.5 * np.pi
    frac_points = 0.9  # Fraction of points to select

    in_samps = 2 ** 10
    out_samps = 2 ** 20

    np.random.seed(1234)
    r = np.random.rand(in_samps)
    x = np.linspace(0.01, 10 * np.pi, in_samps)
    x = x[r >= frac_points]
    y = A * np.cos(w * x + phi)
    f = np.linspace(0.01, 10, out_samps)

    # Use float32 else float64
    if dtype == 'float32':
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        f = f.astype(np.float32)

    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_f = cuda.to_device(f)

    # Run baseline with scipy.signal.lombscargle
    with prof.time_range("scipy_lombscargle", 0):
        cpu_lombscargle = signal.lombscargle(x, y, f)

    # Run Numba version
    with prof.time_range("numba_lombscargle", 1):
        gpu_lombscargle = lombscargle(d_x, d_y, d_f)

    # Copy result to host
    gpu_lombscargle = gpu_lombscargle.copy_to_host()

    # Compare results
    np.testing.assert_allclose(cpu_lombscargle, gpu_lombscargle, 1e-3)    

    # Run multiple passes to get average
    for _ in range(loops):
        with prof.time_range("numba_lombscargle_loop", 2):
            gpu_lombscargle = lombscargle(d_x, d_y, d_f)
