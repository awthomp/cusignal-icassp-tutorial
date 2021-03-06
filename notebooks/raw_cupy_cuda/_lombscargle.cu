
// Copyright (c) 2019-2020, NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// https://github.com/awthomp/cusignal-icassp-tutorial/blob/main/notebooks/raw_cupy_cuda/_lombscargle.cu

///////////////////////////////////////////////////////////////////////////////
//                            LOMBSCARGLE                                    //
///////////////////////////////////////////////////////////////////////////////

// Build
/* 
  nvcc --fatbin -std=c++11 --use_fast_math \
    --generate-code arch=compute_35,code=sm_35 \
    --generate-code arch=compute_35,code=sm_37 \
    --generate-code arch=compute_50,code=sm_50 \
    --generate-code arch=compute_50,code=sm_52 \
    --generate-code arch=compute_53,code=sm_53 \
    --generate-code arch=compute_60,code=sm_60 \
    --generate-code arch=compute_61,code=sm_61 \
    --generate-code arch=compute_62,code=sm_62 \
    --generate-code arch=compute_70,code=sm_70 \
    --generate-code arch=compute_72,code=sm_72 \
	--generate-code arch=compute_75,code=sm_75 \
	--generate-code arch=compute_80,code=sm_80 \
    --generate-code arch=compute_86,code=[sm_86,compute_86] \
    _lombscargle.cu -odir .
*/

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
