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

import numpy as np
import sys

from cupy import prof
from scipy import signal

if __name__ == "__main__":

    dtype = sys.argv[1]
    loops = int(sys.argv[2])

    A = 2.0
    w = 1.0
    phi = 0.5 * np.pi
    frac_points = 0.9  # Fraction of points to select

    in_samps = 2 ** 16
    out_samps = 2 ** 20

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

    with prof.time_range("scipy_lombscargle", 0):
        cpu_lombscargle = signal.lombscargle(x, y, f)

    # Run baseline with scipy.signal.lombscargle
    for _ in range(loops):
        with prof.time_range("scipy_lombscargle_loop", 0):
            cpu_lombscargle = signal.lombscargle(x, y, f)
