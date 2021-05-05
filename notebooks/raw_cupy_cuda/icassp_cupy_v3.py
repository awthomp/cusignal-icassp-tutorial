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
from scipy import signal


# CuPy: Version 3
# Implementations a user level cache from version 2
# and seperates 32 bit and 64 bit versions to
# reduce register pressure.


_kernel_cache = {}


_cupy_lombscargle_src = r"""
extern "C" {
    __global__ void _cupy_lombscargle_float32(
            const int x_shape,
            const int freqs_shape,
            const float * __restrict__ x,
            const float * __restrict__ y,
            const float * __restrict__ freqs,
            float * __restrict__ pgram,
            const float * __restrict__ y_dot
            ) {

        const int tx {
            static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

        float yD {};
        if ( y_dot[0] == 0 ) {
            yD = 1.0f;
        } else {
            yD = 2.0f / y_dot[0];
        }

        for ( int tid = tx; tid < freqs_shape; tid += stride ) {
            float freq { freqs[tid] };
            float xc {};
            float xs {};
            float cc {};
            float ss {};
            float cs {};
            float c {};
            float s {};

            for ( int j = 0; j < x_shape; j++ ) {
                c = cosf( freq * x[j] );
                s = sinf( freq * x[j] );
                xc += y[j] * c;
                xs += y[j] * s;
                cc += c * c;
                ss += s * s;
                cs += c * s;
            }

            float tau { atan2f( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) };
            float c_tau { cosf(freq * tau) };
            float s_tau { sinf(freq * tau) };
            float c_tau2 { c_tau * c_tau };
            float s_tau2 { s_tau * s_tau };
            float cs_tau { 2.0f * c_tau * s_tau };

            pgram[tid] = (
                0.5f * (
                   (
                       ( c_tau * xc + s_tau * xs )
                       * ( c_tau * xc + s_tau * xs )
                       / ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss )
                    )
                   + (
                       ( c_tau * xs - s_tau * xc )
                       * ( c_tau * xs - s_tau * xc )
                       / ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc )
                    )
                )
            ) * yD;
        }
    }

    __global__ void _cupy_lombscargle_float64(
            const int x_shape,
            const int freqs_shape,
            const double * __restrict__ x,
            const double * __restrict__ y,
            const double * __restrict__ freqs,
            double * __restrict__ pgram,
            const double * __restrict__ y_dot
            ) {

        const int tx {
            static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

        double yD {};
        if ( y_dot[0] == 0 ) {
            yD = 1.0;
        } else {
            yD = 2.0 / y_dot[0];
        }

        for ( int tid = tx; tid < freqs_shape; tid += stride ) {

            double freq { freqs[tid] };
            double xc {};
            double xs {};
            double cc {};
            double ss {};
            double cs {};
            double c {};
            double s {};

            for ( int j = 0; j < x_shape; j++ ) {
                c = cos( freq * x[j] );
                s = sin( freq * x[j] );
                xc += y[j] * c;
                xs += y[j] * s;
                cc += c * c;
                ss += s * s;
                cs += c * s;
            }

            double tau { atan2( 2.0 * cs, cc - ss ) / ( 2.0 * freq ) };
            double c_tau { cos(freq * tau) };
            double s_tau { sin(freq * tau) };
            double c_tau2 { c_tau * c_tau };
            double s_tau2 { s_tau * s_tau };
            double cs_tau { 2.0 * c_tau * s_tau };

            pgram[tid] = (
                0.5 * (
                   (
                       ( c_tau * xc + s_tau * xs )
                       * ( c_tau * xc + s_tau * xs )
                       / ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss )
                    )
                   + (
                       ( c_tau * xs - s_tau * xc )
                       * ( c_tau * xs - s_tau * xc )
                       / ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc )
                    )
                )
            ) * yD;
        }
    }
}
"""


def _lombscargle(x, y, freqs, pgram, y_dot):

    if (str(pgram.dtype)) in _kernel_cache:
        kernel = _kernel_cache[(str(pgram.dtype))]
    else:
        module = cp.RawModule(
            code=_cupy_lombscargle_src, options=("-std=c++11",)
        )
        kernel = _kernel_cache[(str(pgram.dtype))] = module.get_function(
            "_cupy_lombscargle_" + str(pgram.dtype)
        )
        print("Registers", kernel.num_regs)

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    threadsperblock = (128,)
    blockspergrid = (numSM * 20,)

    kernel_args = (x.shape[0], freqs.shape[0], x, y, freqs, pgram, y_dot)

    kernel(blockspergrid, threadsperblock, kernel_args)

    cp.cuda.runtime.deviceSynchronize()


def lombscargle(x, y, freqs, precenter=False, normalize=False):

    x = cp.asarray(x)
    y = cp.asarray(y)
    freqs = cp.asarray(freqs)
    pgram = cp.empty(freqs.shape[0], dtype=freqs.dtype)

    assert x.ndim == 1
    assert y.ndim == 1
    assert freqs.ndim == 1

    # Check input sizes
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")

    y_dot = cp.zeros(1, dtype=y.dtype)
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
    check = int(sys.argv[3])

    A = 2.0
    w = 1.0
    phi = 0.5 * np.pi
    frac_points = 0.9  # Fraction of points to select

    in_samps = 2 ** 16
    out_samps = 2 ** 22

    np.random.seed(1234)
    r = np.random.rand(in_samps)
    x = np.linspace(0.01, 10 * np.pi, in_samps)
    x = x[r >= frac_points]
    y = A * np.cos(w * x + phi)
    f = np.linspace(0.01, 10, out_samps)

    # Use float32 else float64
    if dtype == "float32":
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        f = f.astype(np.float32)

    d_x = cp.array(x)
    d_y = cp.array(y)
    d_f = cp.array(f)

    

    # Run Numba version
    with prof.time_range("cupy_lombscargle", 0):
        gpu_lombscargle = lombscargle(d_x, d_y, d_f)

    if check:
        # Run baseline with scipy.signal.lombscargle
        with prof.time_range("scipy_lombscargle", 1):
            cpu_lombscargle = signal.lombscargle(x, y, f)

        # Copy result to host
        gpu_lombscargle = cp.asnumpy(gpu_lombscargle)

        # Compare results
        np.testing.assert_allclose(cpu_lombscargle, gpu_lombscargle, 1e-3)

    # Run multiple passes to get average
    for _ in range(loops):
        with prof.time_range("cupy_lombscargle_loop", 2):
            gpu_lombscargle = lombscargle(d_x, d_y, d_f)