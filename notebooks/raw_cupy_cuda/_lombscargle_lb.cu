
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
    --generate-code arch=compute_75,code=[sm_75,compute_75] \
    _lombscargle_lb.cu -odir .
*/

template<typename T>
__device__ void _cupy_lombscargle_float( const int x_shape,
                                         const int freqs_shape,
                                         const T *__restrict__ x,
                                         const T *__restrict__ y,
                                         const T *__restrict__ freqs,
                                         T *__restrict__ pgram,
                                         const T *__restrict__ y_dot ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    T yD {};
    if ( y_dot[0] == 0 ) {
        yD = 1.0f;
    } else {
        yD = 2.0f / y_dot[0];
    }

    for ( int tid = tx; tid < freqs_shape; tid += stride ) {

        T freq { freqs[tid] };

        T xc {};
        T xs {};
        T cc {};
        T ss {};
        T cs {};
        T c {};
        T s {};

        for ( int j = 0; j < x_shape; j++ ) {
            c = cosf( freq * x[j] );
            s = sinf( freq * x[j] );

            xc += y[j] * c;
            xs += y[j] * s;
            cc += c * c;
            ss += s * s;
            cs += c * s;
        }

        T tau { atan2f( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) };
        T c_tau { cosf( freq * tau ) };
        T s_tau { sinf( freq * tau ) };
        T c_tau2 { c_tau * c_tau };
        T s_tau2 { s_tau * s_tau };
        T cs_tau { 2.0f * c_tau * s_tau };

        pgram[tid] = ( 0.5f * ( ( ( c_tau * xc + s_tau * xs ) * ( c_tau * xc + s_tau * xs ) /
                                  ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
                                ( ( c_tau * xs - s_tau * xc ) * ( c_tau * xs - s_tau * xc ) /
                                  ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) *
                     yD;
    }
}

template<typename T>
__device__ void _cupy_lombscargle_double( const int x_shape,
                                         const int freqs_shape,
                                         const T *__restrict__ x,
                                         const T *__restrict__ y,
                                         const T *__restrict__ freqs,
                                         T *__restrict__ pgram,
                                         const T *__restrict__ y_dot ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    T yD {};
    if ( y_dot[0] == 0 ) {
        yD = 1.0;
    } else {
        yD = 2.0 / y_dot[0];
    }

    for ( int tid = tx; tid < freqs_shape; tid += stride ) {

        T freq { freqs[tid] };

        T xc {};
        T xs {};
        T cc {};
        T ss {};
        T cs {};
        T c {};
        T s {};

        for ( int j = 0; j < x_shape; j++ ) {
            c = cos( freq * x[j] );
            s = sin( freq * x[j] );

            xc += y[j] * c;
            xs += y[j] * s;
            cc += c * c;
            ss += s * s;
            cs += c * s;
        }

        T tau { atan2( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) };
        T c_tau { cos( freq * tau ) };
        T s_tau { sin( freq * tau ) };
        T c_tau2 { c_tau * c_tau };
        T s_tau2 { s_tau * s_tau };
        T cs_tau { 2.0 * c_tau * s_tau };

        pgram[tid] = ( 0.5 * ( ( ( c_tau * xc + s_tau * xs ) * ( c_tau * xc + s_tau * xs ) /
                                  ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
                                ( ( c_tau * xs - s_tau * xc ) * ( c_tau * xs - s_tau * xc ) /
                                  ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) *
                     yD;
    }
}

extern "C" __global__ void __launch_bounds__( 128 ) _cupy_lombscargle_float32( const int x_shape,
                                                                               const int freqs_shape,
                                                                               const float *__restrict__ x,
                                                                               const float *__restrict__ y,
                                                                               const float *__restrict__ freqs,
                                                                               float *__restrict__ pgram,
                                                                               const float *__restrict__ y_dot ) {
    _cupy_lombscargle_float<float>( x_shape, freqs_shape, x, y, freqs, pgram, y_dot );
}

extern "C" __global__ void __launch_bounds__( 128 ) _cupy_lombscargle_float64( const int x_shape,
                                                                               const int freqs_shape,
                                                                               const double *__restrict__ x,
                                                                               const double *__restrict__ y,
                                                                               const double *__restrict__ freqs,
                                                                               double *__restrict__ pgram,
                                                                               const double *__restrict__ y_dot ) {
    _cupy_lombscargle_double<double>( x_shape, freqs_shape, x, y, freqs, pgram, y_dot );
}
