from typing import Literal, Optional, Tuple, cast

from superfv.axes import DIM_TO_AXIS
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, merge_slices, replace_slice

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    lr_conservative_interpolation_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void lr_conservative_interpolation_kernel(
            const double* __restrict__ u,
            double* __restrict__ uj,
            const int p,
            const int dim,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // u    shape (nvars, nx, ny, nz)
            // uj   shape (nvars, nx, ny, nz, 2)
            // p    polynomial degree {0, ..., 7}
            // dim  dimension to interpolate along {0, 1, 2} for (x, y, z, respectively)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ntotal = (long long)nvars * nx * ny * nz;

            double wl0 = 0.0, wl1 = 0.0, wl2 = 0.0, wl3 = 0.0, wl4 = 0.0, wl5 = 0.0,
                wl6 = 0.0, wl7 = 0.0, wl8 = 0.0;
            double wr0 = 0.0, wr1 = 0.0, wr2 = 0.0, wr3 = 0.0, wr4 = 0.0, wr5 = 0.0,
                wr6 = 0.0, wr7 = 0.0, wr8 = 0.0;

            int quadsize;
            switch (p) {
                case 0:
                    wl0 = 1.0;
                    wr0 = 1.0;

                    quadsize = 1;
                    break;
                case 1:
                    wl0 = 1.0/4.0, wl1 = 1.0, wl2 = -1.0/4.0;
                    wr0 = wl2, wr1 = wl1, wr2 = wl0;

                    quadsize = 3;
                    break;
                case 2:
                    wl0 = 1.0/3.0, wl1 = 5.0/6.0, wl2 = -1.0/6.0;
                    wr0 = wl2, wr1 = wl1, wr2 = wl0;

                    quadsize = 3;
                    break;
                case 3:
                    wl0 = -1.0/24.0, wl1 = 5.0/12.0, wl2 = 5.0/6.0, wl3 = -1.0/4.0,
                        wl4 = 1.0/24.0;
                    wr0 = wl4, wr1 = wl3, wr2 = wl2, wr3 = wl1, wr4 = wl0;

                    quadsize = 5;
                    break;
                case 4:
                    wl0 = -1.0/20.0, wl1 = 9.0/20.0, wl2 = 47.0/60.0, wl3 = -13.0/60.0,
                        wl4 = 1.0/30.0;
                    wr0 = wl4, wr1 = wl3, wr2 = wl2, wr3 = wl1, wr4 = wl0;

                    quadsize = 5;
                    break;
                case 5:
                    wl0 = 1.0/120.0, wl1 = -1.0/12.0, wl2 = 59.0/120.0,
                        wl3 = 47.0/60.0, wl4 = -31.0/120.0, wl5 = 1.0/15.0,
                        wl6 = -1.0/120.0;
                    wr0 = wl6, wr1 = wl5, wr2 = wl4, wr3 = wl3, wr4 = wl2, wr5 = wl1,
                        wr6 = wl0;

                    quadsize = 7;
                    break;
                case 6:
                    wl0 = 1.0/105.0, wl1 = -19.0/210.0, wl2 = 107.0/210.0,
                        wl3 = 319.0/420.0, wl4 = -101.0/420.0, wl5 = 5.0/84.0,
                        wl6 = -1.0/140.0;
                    wr0 = wl6, wr1 = wl5, wr2 = wl4, wr3 = wl3, wr4 = wl2, wr5 = wl1,
                        wr6 = wl0;

                    quadsize = 7;
                    break;
                case 7:
                    wl0 = -1.0/560.0, wl1 = 17.0/840.0, wl2 = -97.0/840.0,
                        wl3 = 449.0/840.0, wl4 = 319.0/420.0, wl5 = -223.0/840.0,
                        wl6 = 71.0/840.0, wl7 = -1.0/56.0, wl8 = 1.0/560.0;
                    wr0 = wl8, wr1 = wl7, wr2 = wl6, wr3 = wl5, wr4 = wl4, wr5 = wl3,
                        wr6 = wl2, wr7 = wl1, wr8 = wl0;

                    quadsize = 9;
                    break;
                default:
                    // higher-order interpolation not implemented
                    return; // early return if p is not supported
            }

            const int reach = (quadsize - 1) / 2;

            for (long long i = tid; i < ntotal; i += stride) {
                long long t = i;
                int iz = t % nz; t /= nz;
                int iy = t % ny; t /= ny;
                int ix = t % nx; t /= nx;
                int iv = t;

                // skip threads which reach out of bounds
                switch (dim) {
                    case 0: if (ix < reach || ix >= nx - reach) continue; break;
                    case 1: if (iy < reach || iy >= ny - reach) continue; break;
                    case 2: if (iz < reach || iz >= nz - reach) continue; break;
                    default: return; // invalid dimension
                }

                double result_left = 0.0;
                double result_right = 0.0;
                for (int qj = 0; qj < quadsize; qj++) {
                    // compute neighbor index
                    int off = qj - reach;
                    int jv = iv, jx = ix, jy = iy, jz = iz;
                    switch (dim) {
                        case 0: jx += off; break;
                        case 1: jy += off; break;
                        case 2: jz += off; break;
                    }
                    long long j = ((((long long)jv * nx + jx) * ny + jy) * nz + jz);

                    double wleft = 0.0;
                    double wright = 0.0;
                    switch (qj) {
                        case 0: wleft = wl0; wright = wr0; break;
                        case 1: wleft = wl1; wright = wr1; break;
                        case 2: wleft = wl2; wright = wr2; break;
                        case 3: wleft = wl3; wright = wr3; break;
                        case 4: wleft = wl4; wright = wr4; break;
                        case 5: wleft = wl5; wright = wr5; break;
                        case 6: wleft = wl6; wright = wr6; break;
                        case 7: wleft = wl7; wright = wr7; break;
                        case 8: wleft = wl8; wright = wr8; break;
                        default: continue;
                    }

                    result_left += wleft * u[j];
                    result_right += wright * u[j];
                }

                long long out_idx = ((((long long)iv * nx + ix) * ny + iy) * nz + iz) * 2;
                uj[out_idx] = result_left;
                uj[out_idx + 1] = result_right;
            }
        }

        """,
        name="lr_conservative_interpolation_kernel",
    )

    interpolate_central_quantity_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void interpolate_central_quantity_kernel(
            const double* __restrict__ u,
            double* __restrict__ uj,
            const int mode,
            const int p,
            const int dim,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // u    shape (nvars, nx, ny, nz)
            // uj   shape (nvars, nx, ny, nz)
            // mode 0 for cell-center interpolation, 1 for finite-volume integration
            // p    polynomial degree {0, ..., 7}
            // dim  dimension to interpolate along {0, 1, 2} for (x, y, z, respectively)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            // assign conservative interpolation or transverse integral weights
            double w0 = 0.0, w1 = 0.0, w2 = 0.0, w3 = 0.0, w4 = 0.0, w5 = 0.0, w6 = 0.0;
            int size, reach;
            if (mode == 0) { // interpolate
                if (p <= 1) {
                    w0 = 1.0;
                    size = 1;
                } else if (p <= 3) {
                    w0 = -1.0 / 24.0;
                    w1 = 13.0 / 12.0;
                    w2 = w0;
                    size = 3;
                } else if (p <= 5) {
                    w0 = 3.0 / 640.0;
                    w1 = -29.0 / 480.0;
                    w2 = 1067.0 / 960.0;
                    w3 = w1, w4 = w0;
                    size = 5;
                } else if (p <= 7) {
                    w0 = -5.0 / 7168.0;
                    w1 = 159.0 / 17920;
                    w2 = -7621.0 / 107520.0;
                    w3 = 30251.0 / 26880.0;
                    w4 = w2, w5 = w1, w6 = w0;
                    size = 7;
                } else {
                    // higher-order interpolation not implemented
                    return;
                }
            } else if (mode == 1) { // integrate
                if (p <= 1) {
                    w0 = 1.0;
                    size = 1;
                } else if (p <= 3) {
                    w0 = 1.0 / 24.0;
                    w1 = 11.0 / 12.0;
                    w2 = w0;
                    size = 3;
                } else if (p <= 5) {
                    w0 = -17.0 / 5760.0;
                    w1 = 77.0  / 1440.0;
                    w2 = 863.0 / 960.0;
                    w3 = w1, w4 = w0;
                    size = 5;
                } else if (p <= 7) {
                    w0 = 367.0 / 967680.0;
                    w1 = -281.0 / 53760.0;
                    w2 = 6361.0 / 107520.0;
                    w3 = 215641.0 / 241920.0;
                    w4 = w2, w5 = w1, w6 = w0;
                    size = 7;
                }
            } else {
                return; // invalid mode
            }
            reach = (size - 1) / 2;

            long long ntotal = (long long)nvars*(long long)nx*(long long)ny*(long long)nz;
            for (long long i = tid; i < ntotal; i += stride) {
                long long t = i;
                int iz = t % nz; t /= nz;
                int iy = t % ny; t /= ny;
                int ix = t % nx; t /= nx;
                int iv = t;

                // skip threads which reach out of bounds
                switch (dim) {
                    case 0: if (ix < reach || ix >= nx - reach) continue; break;
                    case 1: if (iy < reach || iy >= ny - reach) continue; break;
                    case 2: if (iz < reach || iz >= nz - reach) continue; break;
                    default: return; // invalid dimension
                }

                double result = 0.0;
                for (int q = 0; q < size; q++) {
                    // get node index j
                    int off = q - reach;
                    int jv = iv;
                    int jx = ix, jy = iy, jz = iz;
                    switch (dim) {
                        case 0: jx += off; break;
                        case 1: jy += off; break;
                        case 2: jz += off; break;
                    }
                    long long j = ((((long long)jv * nx + jx) * ny + jy) * nz + jz);

                    // get stencil weights
                    double w;
                    switch (q) {
                        case 0: w = w0; break;
                        case 1: w = w1; break;
                        case 2: w = w2; break;
                        case 3: w = w3; break;
                        case 4: w = w4; break;
                        case 5: w = w5; break;
                        case 6: w = w6; break;
                    }

                    // update result
                    result += w * u[j];
                }
                // update output array
                uj[i] = result;
            }
        }
        """,
        "interpolate_central_quantity_kernel",
    )

    interpolate_gauss_legendre_nodes_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void interpolate_gauss_legendre_nodes_kernel(
            const double* __restrict__ u,
            double* __restrict__ uj,
            const int p,
            const int dim,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // u    shape (nvars, nx, ny, nz, 2)
            // uj   shape (nvars, nx, ny, nz, 2 * ninterps)
            // p    polynomial degree {0, ..., 7}, determines ninterps

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ntotal = (long long)nvars * nx * ny * nz * 2;

            double x0w0 = 0.0, x0w1 = 0.0, x0w2 = 0.0, x0w3 = 0.0, x0w4 = 0.0;
            double x0w5 = 0.0, x0w6 = 0.0, x0w7 = 0.0, x0w8 = 0.0;
            double x1w0 = 0.0, x1w1 = 0.0, x1w2 = 0.0, x1w3 = 0.0, x1w4 = 0.0;
            double x1w5 = 0.0, x1w6 = 0.0, x1w7 = 0.0, x1w8 = 0.0;
            double x2w0 = 0.0, x2w1 = 0.0, x2w2 = 0.0, x2w3 = 0.0, x2w4 = 0.0;
            double x2w5 = 0.0, x2w6 = 0.0, x2w7 = 0.0, x2w8 = 0.0;
            double x3w0 = 0.0, x3w1 = 0.0, x3w2 = 0.0, x3w3 = 0.0, x3w4 = 0.0;
            double x3w5 = 0.0, x3w6 = 0.0, x3w7 = 0.0, x3w8 = 0.0;

            // assign quadrature weights for each node
            int ninterps, quadsize;
            switch (p) {
                case 0:
                    // first node
                    // u(x=0) = (w_0 * U_0)
                    // w_0: 1
                    x0w0 = 1.0;

                    ninterps = 1, quadsize = 1;
                    break;
                case 1:
                    // first node
                    // u(x=0) = (w_0 * U_0)
                    // w_0: 1
                    x0w0 = 1.0;

                    ninterps = 1, quadsize = 1;
                    break;
                case 2:
                    // first node
                    // u(x=-sqrt(3)*h/6) = (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1)
                    // w_-1: sqrt(3)/12
                    // w_0: 1
                    // w_1: -sqrt(3)/12
                    x0w0 = sqrt(3.0)/12.0, x0w1 = 1.0, x0w2 = -sqrt(3.0)/12.0;

                    // second node
                    // u(x=sqrt(3)*h/6) = (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1)
                    // w_-1: -sqrt(3)/12
                    // w_0: 1
                    // w_1: sqrt(3)/12
                    x1w0 = x0w2, x1w1 = x0w1, x1w2 = x0w0;

                    ninterps = 2, quadsize = 3;
                    break;
                case 3:
                    // first node
                    // u(x=-sqrt(3)*h/6) = (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2)
                    // w_-2: -7*sqrt(3)/432
                    // w_-1: 25*sqrt(3)/216
                    // w_0: 1
                    // w_1: -25*sqrt(3)/216
                    // w_2: 7*sqrt(3)/432
                    x0w0 = -7.0*sqrt(3.0)/432.0, x0w1 = 25.0*sqrt(3.0)/216.0, x0w2 = 1.0;
                    x0w3 = -25.0*sqrt(3.0)/216.0, x0w4 = 7.0*sqrt(3.0)/432.0;

                    // second node
                    // u(x=sqrt(3)*h/6) = (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2)
                    // w_-2: 7*sqrt(3)/432
                    // w_-1: -25*sqrt(3)/216
                    // w_0: 1
                    // w_1: 25*sqrt(3)/216
                    // w_2: -7*sqrt(3)/432
                    x1w0 = x0w4, x1w1 = x0w3, x1w2 = x0w2, x1w3 = x0w1, x1w4 = x0w0;

                    ninterps = 2, quadsize = 5;
                    break;
                case 4:
                    // first node
                    // u(x=-sqrt(15)*h/10) = (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2)
                    // w_-2: -11*sqrt(15)/1200 - 3/800
                    // w_-1: 29/600 + 41*sqrt(15)/600
                    // w_0: 1093/1200
                    // w_1: 29/600 - 41*sqrt(15)/600
                    // w_2: -3/800 + 11*sqrt(15)/1200
                    x0w0 = -11.0*sqrt(15.0)/1200.0 - 3.0/800.0;
                    x0w1 = 29.0/600.0 + 41.0*sqrt(15.0)/600.0;
                    x0w2 = 1093.0/1200.0;
                    x0w3 = 29.0/600.0 - 41.0*sqrt(15.0)/600.0;
                    x0w4 = -3.0/800.0 + 11.0*sqrt(15.0)/1200.0;

                    // second node
                    // u(x=0) = (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2)
                    // w_-2: 3/640
                    // w_-1: -29/480
                    // w_0: 1067/960
                    // w_1: -29/480
                    // w_2: 3/640
                    x1w0 = 3.0/640.0, x1w1 = -29.0/480.0, x1w2 = 1067.0/960.0;
                    x1w3 = x1w1, x1w4 = x1w0;

                    // third node
                    // u(x=sqrt(15)*h/10) = (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2)
                    // w_-2: -3/800 + 11*sqrt(15)/1200
                    // w_-1: 29/600 - 41*sqrt(15)/600
                    // w_0: 1093/1200
                    // w_1: 29/600 + 41*sqrt(15)/600
                    // w_2: -11*sqrt(15)/1200 - 3/800
                    x2w0 = x0w4, x2w1 = x0w3, x2w2 = x0w2, x2w3 = x0w1, x2w4 = x0w0;

                    ninterps = 3, quadsize = 5;
                    break;
                case 5:
                    // first node
                    // u(x=-sqrt(15)*h/10) = (w_-3 * U_-3) + (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2) + (w_3 * U_3)
                    // w_-3: 1363*sqrt(15)/720000
                    // w_-2: -3013*sqrt(15)/180000 - 3/800
                    // w_-1: 29/600 + 11203*sqrt(15)/144000
                    // w_0: 1093/1200
                    // w_1: 29/600 - 11203*sqrt(15)/144000
                    // w_2: -3/800 + 3013*sqrt(15)/180000
                    // w_3: -1363*sqrt(15)/720000
                    x0w0 = 1363.0*sqrt(15.0)/720000.0;
                    x0w1 = -3013.0*sqrt(15.0)/180000.0 - 3.0/800.0;
                    x0w2 = 29.0/600.0 + 11203.0*sqrt(15.0)/144000.0;
                    x0w3 = 1093.0/1200.0;
                    x0w4 = 29.0/600.0 - 11203.0*sqrt(15.0)/144000.0;
                    x0w5 = -3.0/800.0 + 3013.0*sqrt(15.0)/180000.0;
                    x0w6 = -1363.0*sqrt(15.0)/720000.0;

                    // second node
                    // u(x=0) = (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2)
                    // w_-2: 3/640
                    // w_-1: -29/480
                    // w_0: 1067/960
                    // w_1: -29/480
                    // w_2: 3/640
                    x1w0 = 0.0, x1w1 = 3.0/640.0, x1w2 = -29.0/480.0;
                    x1w3 = 1067.0/960.0, x1w4 = x1w2, x1w5 = x1w1, x1w6 = x1w0;

                    // third node
                    // w_-3: -1363*sqrt(15)/720000
                    // w_-2: -3/800 + 3013*sqrt(15)/180000
                    // w_-1: 29/600 - 11203*sqrt(15)/144000
                    // w_0: 1093/1200
                    // w_1: 29/600 + 11203*sqrt(15)/144000
                    // w_2: -3013*sqrt(15)/180000 - 3/800
                    // w_3: 1363*sqrt(15)/720000
                    x2w0 = x0w6, x2w1 = x0w5, x2w2 = x0w4, x2w3 = x0w3;
                    x2w4 = x0w2, x2w5 = x0w1, x2w6 = x0w0;

                    ninterps = 3, quadsize = 7;
                    break;
                case 6:
                    // first node
                    // u(x=-h*sqrt(70*sqrt(30) + 525)/70) = (w_-3 * U_-3) + (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2) + (w_3 * U_3)
                    // w_-3: -59*sqrt(30)*sqrt(70*sqrt(30) + 525)/12348000 + 307/1646400 + 307*sqrt(30)/2744000 + 7039*sqrt(70*sqrt(30) + 525)/24696000
                    // w_-2: -15439*sqrt(70*sqrt(30) + 525)/6174000 - 1971*sqrt(30)/1372000 - 657/274400 + 223*sqrt(30)*sqrt(70*sqrt(30) + 525)/6174000
                    // w_-1: -143*sqrt(30)*sqrt(70*sqrt(30) + 525)/2469600 + 6521/329280 + 6521*sqrt(30)/548800 + 55759*sqrt(70*sqrt(30) + 525)/4939200
                    // w_0: 79423/82320 - 2897*sqrt(30)/137200
                    // w_1: -55759*sqrt(70*sqrt(30) + 525)/4939200 + 143*sqrt(30)*sqrt(70*sqrt(30) + 525)/2469600 + 6521/329280 + 6521*sqrt(30)/548800
                    // w_2: -1971*sqrt(30)/1372000 - 223*sqrt(30)*sqrt(70*sqrt(30) + 525)/6174000 - 657/274400 + 15439*sqrt(70*sqrt(30) + 525)/6174000
                    // w_3: -7039*sqrt(70*sqrt(30) + 525)/24696000 + 307/1646400 + 307*sqrt(30)/2744000 + 59*sqrt(30)*sqrt(70*sqrt(30) + 525)/12348000

                    x0w0 = -59.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/12348000.0
                        + 307.0/1646400.0 + 307.0*sqrt(30.0)/2744000.0
                        + 7039.0*sqrt(70.0*sqrt(30.0) + 525.0)/24696000.0;
                    x0w1 = -15439.0*sqrt(70.0*sqrt(30.0) + 525.0)/6174000.0
                        - 1971.0*sqrt(30.0)/1372000.0 - 657.0/274400.0
                        + 223.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/6174000.0;
                    x0w2 = -143.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/2469600.0
                        + 6521.0/329280.0 + 6521.0*sqrt(30.0)/548800.0
                        + 55759.0*sqrt(70.0*sqrt(30.0) + 525.0)/4939200.0;
                    x0w3 = 79423.0/82320.0 - 2897.0*sqrt(30.0)/137200.0;
                    x0w4 = -55759.0*sqrt(70.0*sqrt(30.0) + 525.0)/4939200.0
                        + 143.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/2469600.0
                        + 6521.0/329280.0 + 6521.0*sqrt(30.0)/548800.0;
                    x0w5 = -1971.0*sqrt(30.0)/1372000.0
                        - 223.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/6174000.0
                        - 657.0/274400.0
                        + 15439.0*sqrt(70.0*sqrt(30.0) + 525.0)/6174000.0;
                    x0w6 = -7039.0*sqrt(70.0*sqrt(30.0) + 525.0)/24696000.0
                        + 307.0/1646400.0 + 307.0*sqrt(30.0)/2744000.0
                        + 59.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/12348000.0;

                    // second node
                    // u(x=-h*sqrt(525 - 70*sqrt(30))/70) = (w_-3 * U_-3) + (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2) + (w_3 * U_3)
                    // w_-3: -307*sqrt(30)/2744000 + 307/1646400 + 59*sqrt(30)*sqrt(525 - 70*sqrt(30))/12348000 + 7039*sqrt(525 - 70*sqrt(30))/24696000
                    // w_-2: -15439*sqrt(525 - 70*sqrt(30))/6174000 - 657/274400 - 223*sqrt(30)*sqrt(525 - 70*sqrt(30))/6174000 + 1971*sqrt(30)/1372000
                    // w_-1: -6521*sqrt(30)/548800 + 143*sqrt(30)*sqrt(525 - 70*sqrt(30))/2469600 + 6521/329280 + 55759*sqrt(525 - 70*sqrt(30))/4939200
                    // w_0: 2897*sqrt(30)/137200 + 79423/82320
                    // w_1: -55759*sqrt(525 - 70*sqrt(30))/4939200 - 6521*sqrt(30)/548800 - 143*sqrt(30)*sqrt(525 - 70*sqrt(30))/2469600 + 6521/329280
                    // w_2: -657/274400 + 223*sqrt(30)*sqrt(525 - 70*sqrt(30))/6174000 + 1971*sqrt(30)/1372000 + 15439*sqrt(525 - 70*sqrt(30))/6174000
                    // w_3: -7039*sqrt(525 - 70*sqrt(30))/24696000 - 307*sqrt(30)/2744000 - 59*sqrt(30)*sqrt(525 - 70*sqrt(30))/12348000 + 307/1646400
                    x1w0 = -307.0*sqrt(30.0)/2744000.0 + 307.0/1646400.0
                        + 59.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/12348000.0
                        + 7039.0*sqrt(525.0 - 70.0*sqrt(30.0))/24696000.0;
                    x1w1 = -15439.0*sqrt(525.0 - 70.0*sqrt(30.0))/6174000.0
                        - 657.0/274400.0
                        - 223.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/6174000.0
                        + 1971.0*sqrt(30.0)/1372000.0;
                    x1w2 = -6521.0*sqrt(30.0)/548800.0 + 143.0*sqrt(30.0)*sqrt(525.0
                        - 70.0*sqrt(30.0))/2469600.0 + 6521.0/329280.0
                        + 55759.0*sqrt(525.0 - 70.0*sqrt(30.0))/4939200.0;
                    x1w3 = 2897.0*sqrt(30.0)/137200.0 + 79423.0/82320.0;
                    x1w4 = -55759.0*sqrt(525.0 - 70.0*sqrt(30.0))/4939200.0
                        - 6521.0*sqrt(30.0)/548800.0
                        - 143.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/2469600.0
                        + 6521.0/329280.0;
                    x1w5 = -657.0/274400.0
                        + 223.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/6174000.0
                        + 1971.0*sqrt(30.0)/1372000.0
                        + 15439.0*sqrt(525.0 - 70.0*sqrt(30.0))/6174000.0;
                    x1w6 = -7039.0*sqrt(525.0 - 70.0*sqrt(30.0))/24696000.0
                        - 307.0*sqrt(30.0)/2744000.0
                        - 59.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/12348000.0
                        + 307.0/1646400.0;

                    // third node
                    // u(x=h*sqrt(525 - 70*sqrt(30))/70) = (w_-3 * U_-3) + (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2) + (w_3 * U_3)
                    // w_-3: -7039*sqrt(525 - 70*sqrt(30))/24696000 - 307*sqrt(30)/2744000 - 59*sqrt(30)*sqrt(525 - 70*sqrt(30))/12348000 + 307/1646400
                    // w_-2: -657/274400 + 223*sqrt(30)*sqrt(525 - 70*sqrt(30))/6174000 + 1971*sqrt(30)/1372000 + 15439*sqrt(525 - 70*sqrt(30))/6174000
                    // w_-1: -55759*sqrt(525 - 70*sqrt(30))/4939200 - 6521*sqrt(30)/548800 - 143*sqrt(30)*sqrt(525 - 70*sqrt(30))/2469600 + 6521/329280
                    // w_0: 2897*sqrt(30)/137200 + 79423/82320
                    // w_1: -6521*sqrt(30)/548800 + 143*sqrt(30)*sqrt(525 - 70*sqrt(30))/2469600 + 6521/329280 + 55759*sqrt(525 - 70*sqrt(30))/4939200
                    // w_2: -15439*sqrt(525 - 70*sqrt(30))/6174000 - 657/274400 - 223*sqrt(30)*sqrt(525 - 70*sqrt(30))/6174000 + 1971*sqrt(30)/1372000
                    // w_3: -307*sqrt(30)/2744000 + 307/1646400 + 59*sqrt(30)*sqrt(525 - 70*sqrt(30))/12348000 + 7039*sqrt(525 - 70*sqrt(30))/24696000
                    x2w0 = x1w6, x2w1 = x1w5, x2w2 = x1w4, x2w3 = x1w3, x2w4 = x1w2;
                    x2w5 = x1w1, x2w6 = x1w0;

                    // fourth node
                    // u(x=h*sqrt(70*sqrt(30) + 525)/70) = (w_-3 * U_-3) + (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2) + (w_3 * U_3)
                    // w_-3: -7039*sqrt(70*sqrt(30) + 525)/24696000 + 307/1646400 + 307*sqrt(30)/2744000 + 59*sqrt(30)*sqrt(70*sqrt(30) + 525)/12348000
                    // w_-2: -1971*sqrt(30)/1372000 - 223*sqrt(30)*sqrt(70*sqrt(30) + 525)/6174000 - 657/274400 + 15439*sqrt(70*sqrt(30) + 525)/6174000
                    // w_-1: -55759*sqrt(70*sqrt(30) + 525)/4939200 + 143*sqrt(30)*sqrt(70*sqrt(30) + 525)/2469600 + 6521/329280 + 6521*sqrt(30)/548800
                    // w_0: 79423/82320 - 2897*sqrt(30)/137200
                    // w_1: -143*sqrt(30)*sqrt(70*sqrt(30) + 525)/2469600 + 6521/329280 + 6521*sqrt(30)/548800 + 55759*sqrt(70*sqrt(30) + 525)/4939200
                    // w_2: -15439*sqrt(70*sqrt(30) + 525)/6174000 - 1971*sqrt(30)/1372000 - 657/274400 + 223*sqrt(30)*sqrt(70*sqrt(30) + 525)/6174000
                    // w_3: -59*sqrt(30)*sqrt(70*sqrt(30) + 525)/12348000 + 307/1646400 + 307*sqrt(30)/2744000 + 7039*sqrt(70*sqrt(30) + 525)/24696000
                    x3w0 = x0w6, x3w1 = x0w5, x3w2 = x0w4, x3w3 = x0w3, x3w4 = x0w2;
                    x3w5 = x0w1, x3w6 = x0w0;

                    ninterps = 4, quadsize = 7;
                    break;
                case 7:
                    // first node
                    // u(x=-h*sqrt(70*sqrt(30) + 525)/70) = (w_-4 * U_-4) + (w_-3 * U_-3) + (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2) + (w_3 * U_3) + (w_4 * U_4)
                    // w_-4: -37831*sqrt(70*sqrt(30) + 525)/605052000 + 3177*sqrt(30)*sqrt(70*sqrt(30) + 525)/2689120000
                    // w_-3: -143599*sqrt(30)*sqrt(70*sqrt(30) + 525)/12101040000 + 307/1646400 + 307*sqrt(30)/2744000 + 798883*sqrt(70*sqrt(30) + 525)/1210104000
                    // w_-2: -9119*sqrt(70*sqrt(30) + 525)/2701125 - 1971*sqrt(30)/1372000 - 657/274400 + 91033*sqrt(30)*sqrt(70*sqrt(30) + 525)/1728720000
                    // w_-1: -128693*sqrt(30)*sqrt(70*sqrt(30) + 525)/1728720000 + 6521/329280 + 6521*sqrt(30)/548800 + 700963*sqrt(70*sqrt(30) + 525)/57624000
                    // w_0: 79423/82320 - 2897*sqrt(30)/137200
                    // w_1: -700963*sqrt(70*sqrt(30) + 525)/57624000 + 128693*sqrt(30)*sqrt(70*sqrt(30) + 525)/1728720000 + 6521/329280 + 6521*sqrt(30)/548800
                    // w_2: -91033*sqrt(30)*sqrt(70*sqrt(30) + 525)/1728720000 - 1971*sqrt(30)/1372000 - 657/274400 + 9119*sqrt(70*sqrt(30) + 525)/2701125
                    // w_3: -798883*sqrt(70*sqrt(30) + 525)/1210104000 + 307/1646400 + 307*sqrt(30)/2744000 + 143599*sqrt(30)*sqrt(70*sqrt(30) + 525)/12101040000
                    // w_4: -3177*sqrt(30)*sqrt(70*sqrt(30) + 525)/2689120000 + 37831*sqrt(70*sqrt(30) + 525)/605052000
                    x0w0 = -37831.0*sqrt(70.0*sqrt(30.0) + 525.0)/605052000.0
                        + 3177.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/2689120000.0;
                    x0w1 = -143599.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/12101040000.0
                        + 307.0/1646400.0 + 307.0*sqrt(30.0)/2744000.0
                        + 798883.0*sqrt(70.0*sqrt(30.0) + 525.0)/1210104000.0;
                    x0w2 = -9119.0*sqrt(70.0*sqrt(30.0) + 525.0)/2701125.0
                        - 1971.0*sqrt(30.0)/1372000.0 - 657.0/274400.0
                        + 91033.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/1728720000.0;
                    x0w3 = -128693.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/1728720000.0
                        + 6521.0/329280.0 + 6521.0*sqrt(30.0)/548800.0
                        + 700963.0*sqrt(70.0*sqrt(30.0) + 525.0)/57624000.0;
                    x0w4 = 79423.0/82320.0 - 2897.0*sqrt(30.0)/137200.0;
                    x0w5 = -700963.0*sqrt(70.0*sqrt(30.0) + 525.0)/57624000.0
                        + 128693.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/1728720000.0
                        + 6521.0/329280.0 + 6521.0*sqrt(30.0)/548800.0;
                    x0w6 = -91033.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/1728720000.0
                        - 1971.0*sqrt(30.0)/1372000.0 - 657.0/274400.0
                        + 9119.0*sqrt(70.0*sqrt(30.0) + 525.0)/2701125.0;
                    x0w7 = -798883.0*sqrt(70.0*sqrt(30.0) + 525.0)/1210104000.0
                        + 307.0/1646400.0 + 307.0*sqrt(30.0)/2744000.0
                        + 143599.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/12101040000.0;
                    x0w8 = -3177.0*sqrt(30.0)*sqrt(70.0*sqrt(30.0) + 525.0)/2689120000.0
                        + 37831.0*sqrt(70.0*sqrt(30.0) + 525.0)/605052000.0;

                    // second node
                    // u(x=-h*sqrt(525 - 70*sqrt(30))/70) = (w_-4 * U_-4) + (w_-3 * U_-3) + (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2) + (w_3 * U_3) + (w_4 * U_4)
                    // w_-4: -37831*sqrt(525 - 70*sqrt(30))/605052000 - 3177*sqrt(30)*sqrt(525 - 70*sqrt(30))/2689120000
                    // w_-3: -307*sqrt(30)/2744000 + 307/1646400 + 143599*sqrt(30)*sqrt(525 - 70*sqrt(30))/12101040000 + 798883*sqrt(525 - 70*sqrt(30))/1210104000
                    // w_-2: -9119*sqrt(525 - 70*sqrt(30))/2701125 - 91033*sqrt(30)*sqrt(525 - 70*sqrt(30))/1728720000 - 657/274400 + 1971*sqrt(30)/1372000
                    // w_-1: -6521*sqrt(30)/548800 + 128693*sqrt(30)*sqrt(525 - 70*sqrt(30))/1728720000 + 6521/329280 + 700963*sqrt(525 - 70*sqrt(30))/57624000
                    // w_0: 2897*sqrt(30)/137200 + 79423/82320
                    // w_1: -700963*sqrt(525 - 70*sqrt(30))/57624000 - 6521*sqrt(30)/548800 - 128693*sqrt(30)*sqrt(525 - 70*sqrt(30))/1728720000 + 6521/329280
                    // w_2: -657/274400 + 91033*sqrt(30)*sqrt(525 - 70*sqrt(30))/1728720000 + 1971*sqrt(30)/1372000 + 9119*sqrt(525 - 70*sqrt(30))/2701125
                    // w_3: -798883*sqrt(525 - 70*sqrt(30))/1210104000 - 143599*sqrt(30)*sqrt(525 - 70*sqrt(30))/12101040000 - 307*sqrt(30)/2744000 + 307/1646400
                    // w_4: 3177*sqrt(30)*sqrt(525 - 70*sqrt(30))/2689120000 + 37831*sqrt(525 - 70*sqrt(30))/605052000
                    x1w0 = -37831.0*sqrt(525.0 - 70.0*sqrt(30.0))/605052000.0
                        - 3177.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/2689120000.0;
                    x1w1 = -307.0*sqrt(30.0)/2744000.0 + 307.0/1646400.0
                    + 143599.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/12101040000.0
                    + 798883.0*sqrt(525.0 - 70.0*sqrt(30.0))/1210104000.0;
                    x1w2 = -9119.0*sqrt(525.0 - 70.0*sqrt(30.0))/2701125.0
                        - 91033.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/1728720000.0
                        - 657.0/274400.0 + 1971.0*sqrt(30.0)/1372000.0;
                    x1w3 = -6521.0*sqrt(30.0)/548800.0
                        + 128693.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/1728720000.0
                        + 6521.0/329280.0
                        + 700963.0*sqrt(525.0 - 70.0*sqrt(30.0))/57624000.0;
                    x1w4 = 2897.0*sqrt(30.0)/137200.0 + 79423.0/82320.0;
                    x1w5 = -700963.0*sqrt(525.0 - 70.0*sqrt(30.0))/57624000.0
                        - 6521.0*sqrt(30.0)/548800.0
                        - 128693.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/1728720000.0
                        + 6521.0/329280.0;
                    x1w6 = -657.0/274400.0
                        + 91033.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/1728720000.0
                        + 1971.0*sqrt(30.0)/1372000.0
                        + 9119.0*sqrt(525.0 - 70.0*sqrt(30.0))/2701125.0;
                    x1w7 = -798883.0*sqrt(525.0 - 70.0*sqrt(30.0))/1210104000.0
                        - 143599.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/12101040000.0
                        - 307.0*sqrt(30.0)/2744000.0 + 307.0/1646400.0;
                    x1w8 = 3177.0*sqrt(30.0)*sqrt(525.0 - 70.0*sqrt(30.0))/2689120000.0
                        + 37831.0*sqrt(525.0 - 70.0*sqrt(30.0))/605052000.0;

                    // third node
                    // u(x=h*sqrt(525 - 70*sqrt(30))/70) = (w_-4 * U_-4) + (w_-3 * U_-3) + (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2) + (w_3 * U_3) + (w_4 * U_4)
                    // w_-4: 3177*sqrt(30)*sqrt(525 - 70*sqrt(30))/2689120000 + 37831*sqrt(525 - 70*sqrt(30))/605052000
                    // w_-3: -798883*sqrt(525 - 70*sqrt(30))/1210104000 - 143599*sqrt(30)*sqrt(525 - 70*sqrt(30))/12101040000 - 307*sqrt(30)/2744000 + 307/1646400
                    // w_-2: -657/274400 + 91033*sqrt(30)*sqrt(525 - 70*sqrt(30))/1728720000 + 1971*sqrt(30)/1372000 + 9119*sqrt(525 - 70*sqrt(30))/2701125
                    // w_-1: -700963*sqrt(525 - 70*sqrt(30))/57624000 - 6521*sqrt(30)/548800 - 128693*sqrt(30)*sqrt(525 - 70*sqrt(30))/1728720000 + 6521/329280
                    // w_0: 2897*sqrt(30)/137200 + 79423/82320
                    // w_1: -6521*sqrt(30)/548800 + 128693*sqrt(30)*sqrt(525 - 70*sqrt(30))/1728720000 + 6521/329280 + 700963*sqrt(525 - 70*sqrt(30))/57624000
                    // w_2: -9119*sqrt(525 - 70*sqrt(30))/2701125 - 91033*sqrt(30)*sqrt(525 - 70*sqrt(30))/1728720000 - 657/274400 + 1971*sqrt(30)/1372000
                    // w_3: -307*sqrt(30)/2744000 + 307/1646400 + 143599*sqrt(30)*sqrt(525 - 70*sqrt(30))/12101040000 + 798883*sqrt(525 - 70*sqrt(30))/1210104000
                    // w_4: -37831*sqrt(525 - 70*sqrt(30))/605052000 - 3177*sqrt(30)*sqrt(525 - 70*sqrt(30))/2689120000
                    x2w0 = x1w8, x2w1 = x1w7, x2w2 = x1w6, x2w3 = x1w5, x2w4 = x1w4;
                    x2w5 = x1w3, x2w6 = x1w2, x2w7 = x1w1, x2w8 = x1w0;

                    // fourth node
                    // u(x=h*sqrt(70*sqrt(30) + 525)/70) = (w_-4 * U_-4) + (w_-3 * U_-3) + (w_-2 * U_-2) + (w_-1 * U_-1) + (w_0 * U_0) + (w_1 * U_1) + (w_2 * U_2) + (w_3 * U_3) + (w_4 * U_4)
                    // w_-4: -3177*sqrt(30)*sqrt(70*sqrt(30) + 525)/2689120000 + 37831*sqrt(70*sqrt(30) + 525)/605052000
                    // w_-3: -798883*sqrt(70*sqrt(30) + 525)/1210104000 + 307/1646400 + 307*sqrt(30)/2744000 + 143599*sqrt(30)*sqrt(70*sqrt(30) + 525)/12101040000
                    // w_-2: -91033*sqrt(30)*sqrt(70*sqrt(30) + 525)/1728720000 - 1971*sqrt(30)/1372000 - 657/274400 + 9119*sqrt(70*sqrt(30) + 525)/2701125
                    // w_-1: -700963*sqrt(70*sqrt(30) + 525)/57624000 + 128693*sqrt(30)*sqrt(70*sqrt(30) + 525)/1728720000 + 6521/329280 + 6521*sqrt(30)/548800
                    // w_0: 79423/82320 - 2897*sqrt(30)/137200
                    // w_1: -128693*sqrt(30)*sqrt(70*sqrt(30) + 525)/1728720000 + 6521/329280 + 6521*sqrt(30)/548800 + 700963*sqrt(70*sqrt(30) + 525)/57624000
                    // w_2: -9119*sqrt(70*sqrt(30) + 525)/2701125 - 1971*sqrt(30)/1372000 - 657/274400 + 91033*sqrt(30)*sqrt(70*sqrt(30) + 525)/1728720000
                    // w_3: -143599*sqrt(30)*sqrt(70*sqrt(30) + 525)/12101040000 + 307/1646400 + 307*sqrt(30)/2744000 + 798883*sqrt(70*sqrt(30) + 525)/1210104000
                    // w_4: -37831*sqrt(70*sqrt(30) + 525)/605052000 + 3177*sqrt(30)*sqrt(70*sqrt(30) + 525)/2689120000
                    x3w0 = x0w8, x3w1 = x0w7, x3w2 = x0w6, x3w3 = x0w5, x3w4 = x0w4;
                    x3w5 = x0w3, x3w6 = x0w2, x3w7 = x0w1, x3w8 = x0w0;

                    ninterps = 4, quadsize = 9;
                    break;
                default:
                    // unsupported quadrature order
                    return;
            }
            const int reach = (quadsize - 1) / 2;

            for (long long i = tid; i < ntotal; i += stride) {
                long long t = i;
                int iface = t % 2; t /= 2;
                int iz = t % nz; t /= nz;
                int iy = t % ny; t /= ny;
                int ix = t % nx; t /= nx;
                int iv = t % nvars;

                switch (dim) {
                    case 0: if (ix < reach || ix >= nx - reach) continue; break;
                    case 1: if (iy < reach || iy >= ny - reach) continue; break;
                    case 2: if (iz < reach || iz >= nz - reach) continue; break;
                }

                for (int interp_idx = 0; interp_idx < ninterps; interp_idx++) {
                    double result = 0.0;
                    long long output_idx
                        = (((long long)iv * nx + ix) * ny * nz + iy * nz + iz)
                        * (2 * ninterps) + iface * ninterps + interp_idx;

                    for (int qj = 0; qj < quadsize; qj++) {
                        // get neighbor index
                        int offset = qj - reach;
                        int jv = iv, jx = ix, jy = iy, jz = iz;
                        switch (dim) {
                            case 0: jx += offset; break;
                            case 1: jy += offset; break;
                            case 2: jz += offset; break;
                        }
                        long long j = ((((long long)jv * nx + jx) * ny + jy) * nz + jz)
                            * 2 + iface;

                        // get weight
                        double w;
                        switch (interp_idx) {
                            case 0:
                                switch (qj) {
                                    case 0: w = x0w0; break;
                                    case 1: w = x0w1; break;
                                    case 2: w = x0w2; break;
                                    case 3: w = x0w3; break;
                                    case 4: w = x0w4; break;
                                    case 5: w = x0w5; break;
                                    case 6: w = x0w6; break;
                                    case 7: w = x0w7; break;
                                    case 8: w = x0w8; break;
                                } break;
                            case 1:
                                switch (qj) {
                                    case 0: w = x1w0; break;
                                    case 1: w = x1w1; break;
                                    case 2: w = x1w2; break;
                                    case 3: w = x1w3; break;
                                    case 4: w = x1w4; break;
                                    case 5: w = x1w5; break;
                                    case 6: w = x1w6; break;
                                    case 7: w = x1w7; break;
                                    case 8: w = x1w8; break;
                                } break;
                            case 2:
                                switch (qj) {
                                    case 0: w = x2w0; break;
                                    case 1: w = x2w1; break;
                                    case 2: w = x2w2; break;
                                    case 3: w = x2w3; break;
                                    case 4: w = x2w4; break;
                                    case 5: w = x2w5; break;
                                    case 6: w = x2w6; break;
                                    case 7: w = x2w7; break;
                                    case 8: w = x2w8; break;
                                } break;
                            case 3:
                                switch (qj) {
                                    case 0: w = x3w0; break;
                                    case 1: w = x3w1; break;
                                    case 2: w = x3w2; break;
                                    case 3: w = x3w3; break;
                                    case 4: w = x3w4; break;
                                    case 5: w = x3w5; break;
                                    case 6: w = x3w6; break;
                                    case 7: w = x3w7; break;
                                    case 8: w = x3w8; break;
                                } break;
                        }
                        result += w * u[j];
                    }
                    uj[output_idx] = result;
                }
            }
        }
        """,
        name="interpolate_gauss_legendre_nodes_kernel",
    )

    gauss_legendre_quadrature_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void gauss_legendre_quadrature_kernel(
            const double* wj,
            double* out,
            const int nvars,
            const int nx,
            const int ny,
            const int nz,
            const int nquadrature
        ){
            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ntotal = (long long)nvars * nx * ny * nz;

            double w0=0, w1=0, w2=0, w3=0, w4=0;
            switch (nquadrature) {
                case 1:
                    w0 = 1.0;
                    break;
                case 2:
                    w0 = 0.5;
                    w1 = w0;
                    break;
                case 3:
                    w0 = 5.0 / 18.0;
                    w1 = 4.0 / 9.0;
                    w2 = w0;
                    break;
                case 4:
                    w0 = (18.0 - sqrt(30.0)) / 72.0;
                    w1 = (18.0 + sqrt(30.0)) / 72.0;
                    w2 = w1;
                    w3 = w0;
                    break;
                case 5:
                    w0 = (322.0 - 13.0 * sqrt(70.0)) / 1800.0;
                    w1 = (322.0 + 13.0 * sqrt(70.0)) / 1800.0;
                    w2 = 64.0 / 225;
                    w3 = w1;
                    w4 = w0;
                    break;
                default:
                    return;
                }

            for (long long i = tid; i < ntotal; i += stride) {
                const double* row = wj + ((size_t)i * nquadrature);
                double result = 0.0;
                for (int j = 0; j < nquadrature; j++) {
                    double wq = row[j];
                    switch (j) {
                        case 0: result += w0 * wq; break;
                        case 1: result += w1 * wq; break;
                        case 2: result += w2 * wq; break;
                        case 3: result += w3 * wq; break;
                        case 4: result += w4 * wq; break;
                    }
                }
                out[i] = result;
            }
    }
        """,
        name="gauss_legendre_quadrature_kernel",
    )


def lr_conservative_interpolation_kernel_helper(
    u: ArrayLike, out: ArrayLike, p: int, dim: Literal["x", "y", "z"]
) -> Tuple[slice, ...]:
    dim_int = {"x": 0, "y": 1, "z": 2}[dim]

    if p == 0:
        if not (out.ndim == 5 and out.shape[4] == 2):
            raise ValueError(
                "Expected `out` to have 5 dimensions with the last one having "
                "length 2"
            )
        out[..., 0] = u
        out[..., 1] = u
        return (slice(None), slice(None), slice(None), slice(None), slice(None, 2))
    if p not in {1, 2, 3, 4, 5, 6, 7}:
        raise ValueError("Expected `p` in {0,...,7}")

    reach = -(-p // 2)
    nvars, nx, ny, nz = u.shape

    if not (out.ndim == 5 and out.shape[4] == 2):
        raise ValueError(
            "Expected `out` to have 5 dimensions with the last one having " "length 2"
        )
    if out.shape[:4] != u.shape:
        raise ValueError(
            "Expected `out` to have shape (nvars, nx, ny, nz, 2) matching `u`"
        )

    if not u.flags.c_contiguous:
        raise ValueError("Input array must be C-contiguous for CUDA kernel.")
    if not out.flags.c_contiguous:
        raise ValueError("Output array must be C-contiguous for CUDA kernel.")
    if not u.dtype == cp.float64:
        raise ValueError("Input array must be of type float64 for CUDA kernel.")
    if not out.dtype == cp.float64:
        raise ValueError("Output array must be of type float64 for CUDA kernel.")

    threads_per_block = 256
    blocks_per_grid = (
        nvars * nx * ny * nz + threads_per_block - 1
    ) // threads_per_block
    lr_conservative_interpolation_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (u, out, p, dim_int, nvars, nx, ny, nz),
    )

    axis = DIM_TO_AXIS[dim]
    valid = replace_slice(crop(axis, (reach, -reach), ndim=5), 4, slice(None, 2))
    return cast(Tuple[slice, ...], valid)


def interpolate_central_quantity_kernel_helper(
    u: ArrayLike,
    uj: ArrayLike,
    mode: Literal[0, 1],
    p: int,
    dim: Literal["x", "y", "z"],
) -> Tuple[slice, ...]:
    if mode not in {0, 1}:
        raise ValueError("Mode must be 0 for interpolation or 1 for integration")
    if p not in {0, 1, 2, 3, 4, 5, 6, 7}:
        raise ValueError("Polynomial degree p must be an integer in the range [0, 7]")
    if u.ndim != 4:
        raise ValueError("Input array u must have 4 dimensions (nvars, nx, ny, nz)")
    if not (u.shape == uj.shape):
        raise ValueError(
            "Input and output arrays must have the same shape. "
            f"Got {u.shape=} and {uj.shape=}"
        )
    if not u.flags.c_contiguous:
        raise ValueError("Input array u must be C-contiguous")
    if not uj.flags.c_contiguous:
        raise ValueError("Output array uj must be C-contiguous")
    if u.dtype != cp.float64:
        raise ValueError("Input array u must be of type float64")
    if uj.dtype != cp.float64:
        raise ValueError("Output array uj must be of type float64")

    nvars, nx, ny, nz = u.shape
    dim_int = {"x": 0, "y": 1, "z": 2}[dim]
    reach = p // 2

    n = u.size
    threads = 256
    blocks = (n + threads - 1) // threads

    interpolate_central_quantity_kernel(
        (blocks,), (threads,), (u, uj, mode, p, dim_int, nvars, nx, ny, nz)
    )
    return crop(DIM_TO_AXIS[dim], (reach, -reach), ndim=4)


def interpolate_central_quantity(
    u: ArrayLike,
    uj: ArrayLike,
    mode: Literal[0, 1],
    p: int,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    uu: Optional[ArrayLike] = None,
    uuu: Optional[ArrayLike] = None,
) -> Tuple[slice, ...]:
    """
    Perform a central sweep interpolation or integration along the specified
    active dimensions.

    Args:
        u: Input array of shape (nvars, nx, ny, nz) containing the original data.
        uj: Output array of shape (nvars, nx, ny, nz) to store the results of the
            interpolation/integration.
        mode: 0 for cell-center interpolation, 1 for finite-volume integration.
        p: Polynomial degree for the interpolation/integration (0 to 7).
        active_dims: Tuple of active dimensions to perform the sweep along, each being
            "x", "y", or "z". The length of this tuple determines the number of sweeps.
        uu: Optional intermediate array for 2D interpolation with shape
            (nvars, nx, ny, nz).
        uuu: Optional intermediate array for 3D interpolation with shape
            (nvars, nx, ny, nz).

    Returns:
        A tuple of slices corresponding to the valid region of the output array after
        performing the central sweep along the specified dimensions.
    """
    ndim = len(active_dims)

    if ndim == 1:
        return interpolate_central_quantity_kernel_helper(
            u, uj, mode, p, active_dims[0]
        )

    if ndim == 2:
        if uu is None:
            raise ValueError(
                "Intermediate array uu must be provided for 2D interpolation"
            )
        slc1 = interpolate_central_quantity_kernel_helper(
            uu, u, mode, p, active_dims[0]
        )
        slc2 = interpolate_central_quantity_kernel_helper(
            u, uj, mode, p, active_dims[1]
        )
        return merge_slices(slc1, slc2)

    if ndim == 3:
        if uu is None or uuu is None:
            raise ValueError(
                "Intermediate arrays uu and uuu must be provided for 3D interpolation"
            )
        slc1 = interpolate_central_quantity_kernel_helper(
            uuu, uu, mode, p, active_dims[0]
        )
        slc2 = interpolate_central_quantity_kernel_helper(
            uu, u, mode, p, active_dims[1]
        )
        slc3 = interpolate_central_quantity_kernel_helper(
            u, uj, mode, p, active_dims[2]
        )
        return merge_slices(slc1, slc2, slc3)

    raise ValueError("active_dims must have length 1, 2, or 3")


def gauss_legendre_quadrature_kernel_helper(u: ArrayLike, p: int, out: ArrayLike):
    nquadrature = -(-(p + 1) // 2)

    if u.ndim != 5 or u.shape[4] != nquadrature:
        raise ValueError(
            "Expected input `u` to have 5 dimensions with the last one having length "
            f"{nquadrature}"
        )
    if out.ndim != 4 or out.shape != u.shape[:4]:
        raise ValueError(
            "Expected output `out` to have 4 dimensions and the same shape as the first "
            "4 dimensions of `u`"
        )
    if not out.flags.c_contiguous:
        raise ValueError("Output array must be C-contiguous for CUDA kernel.")
    if not u.flags.c_contiguous:
        raise ValueError("Input array must be C-contiguous for CUDA kernel.")
    if not out.dtype == cp.float64:
        raise ValueError("Output array must be of type float64 for CUDA kernel.")
    if not u.dtype == cp.float64:
        raise ValueError("Input array must be of type float64 for CUDA kernel.")

    nvars, nx, ny, nz, _ = u.shape
    threads_per_block = 256
    blocks_per_grid = (u.size + threads_per_block - 1) // threads_per_block
    gauss_legendre_quadrature_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (u[..., :nquadrature], out, nvars, nx, ny, nz, nquadrature),
    )


def interpolate_gauss_legendre_nodes_kernel_helper(
    u: ArrayLike, out: ArrayLike, p: int, dim: Literal["x", "y", "z"]
) -> Tuple[slice, ...]:
    dim_int = {"x": 0, "y": 1, "z": 2}[dim]
    ninterps = p // 2 + 1
    reach = -(-p // 2)

    if p in {0, 1}:
        out[..., :2] = u
        return (slice(None), slice(None), slice(None), slice(None), slice(None, 2))
    if p not in {2, 3, 4, 5, 6, 7}:
        raise ValueError("Expected `p` in {0,...,7}")
    if u.ndim != 5 or u.shape[4] != 2:
        raise ValueError("Expected input `u` with shape (nvars, nx, ny, nz, 2)")
    if out.ndim != 5 or out.shape[4] != 2 * ninterps:
        raise ValueError(
            "Expected `out` to have 5 dimensions with the last one having length "
            f"{2 * ninterps}"
        )
    if out.shape[:4] != u.shape[:4]:
        raise ValueError(
            "Expected `out` to have the same shape as the first 4 dimensions of `u`"
        )
    if not u.flags.c_contiguous:
        raise ValueError("Input array must be C-contiguous for CUDA kernel.")
    if not out.flags.c_contiguous:
        raise ValueError("Output array must be C-contiguous for CUDA kernel.")
    if not u.dtype == cp.float64:
        raise ValueError("Input array must be of type float64 for CUDA kernel.")
    if not out.dtype == cp.float64:
        raise ValueError("Output array must be of type float64 for CUDA kernel.")

    nvars, nx, ny, nz, _ = u.shape

    threads_per_block = 256
    blocks_per_grid = (
        nvars * nx * ny * nz + threads_per_block - 1
    ) // threads_per_block
    interpolate_gauss_legendre_nodes_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (u, out, p, dim_int, nvars, nx, ny, nz),
    )

    axis = DIM_TO_AXIS[dim]
    ninterps_slc = slice(None, 2 * ninterps)
    valid = replace_slice(crop(axis, (reach, -reach), ndim=5), 4, ninterps_slc)
    return cast(Tuple[slice, ...], valid)
