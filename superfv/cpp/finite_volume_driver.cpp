#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "stencil_application.h"

namespace py = pybind11;

int count_ndim(const int nx, const int ny, const int nz) {
    if (!(nx >= ny && ny >= nz)) {
        throw std::invalid_argument("Invalid grid dimensions");
    }
    return static_cast<int>(nx > 1) + static_cast<int>(ny > 1) + static_cast<int>(nz > 1);
}

double _interpolate_cell_center_or_average(
    const double* u_ijk,
    const int p,
    const int nx,
    const int ny,
    const int nz,
    const bool cell_centers
) {
    // u_ijk points to u[..., i, j, k], with u having shape (..., nx, ny, nz)

    const int ndim = count_ndim(nx, ny, nz);
    const int nkernel_max = 7;

    double stencil[nkernel_max] = {0.0};
    int nkernel = 0;
    if (cell_centers) {
        if (p == 0 or p == 1) {
            stencil[0] = 1.0;
            nkernel = 1;
        } else if (p == 2 or p == 3) {
            stencil[0] = -1.0 / 24.0;
            stencil[1] = 13.0 / 12.0;
            stencil[2] = -1.0 / 24.0;
            nkernel = 3;
        } else if (p == 4 or p == 5) {
            stencil[0] = 3.0 / 640.0;
            stencil[1] = -29.0 / 480.0;
            stencil[2] = 1067.0 / 960.0;
            stencil[3] = -29.0 / 480.0;
            stencil[4] = 3.0 / 640.0;
            nkernel = 5;
        } else if (p == 6 or p == 7) {
            stencil[0] = -5.0 / 7168.0;
            stencil[1] = 159.0 / 17920.0;
            stencil[2] = -7621.0 / 107520.0;
            stencil[3] = 30251.0 / 26880.0;
            stencil[4] = -7621.0 / 107520.0;
            stencil[5] = 159.0 / 17920.0;
            stencil[6] = -5.0 / 7168.0;
            nkernel = 7;
        } else {
            throw std::invalid_argument("Invalid order of interpolation");
        }
    } else {
        if (p == 0 or p == 1) {
            stencil[0] = 1.0;
            nkernel = 1;
        } else if (p == 2 or p == 3) {
            stencil[0] = 1.0 / 24.0;
            stencil[1] = 11.0 / 12.0;
            stencil[2] = 1.0 / 24.0;
            nkernel = 3;
        } else if (p == 4 or p == 5) {
            stencil[0] = -17.0 / 5760.0;
            stencil[1] = 77.0 / 1440.0;
            stencil[2] = 863.0 / 960.0;
            stencil[3] = 77.0 / 1440.0;
            stencil[4] = -17.0 / 5760.0;
            nkernel = 5;
        } else if (p == 6 or p == 7) {
            stencil[0] = 367.0 / 967680.0;
            stencil[1] = -281.0 / 53760.0;
            stencil[2] = 6361.0 / 107520.0;
            stencil[3] = 215641.0 / 241920.0;
            stencil[4] = 6361.0 / 107520.0;
            stencil[5] = -281.0 / 53760.0;
            stencil[6] = 367.0 / 967680.0;
            nkernel = 7;
        } else {
            throw std::invalid_argument("Invalid order of interpolation");
        }
    }

    if (ndim == 1) {
        return apply_1d_stencil(u_ijk, stencil, 0, ny, nz, nkernel);
    } else if (ndim == 2) {
        double temp[nkernel_max] = {0.0};
        return apply_2d_stencil(u_ijk, stencil, stencil, temp, 0, 1, ny, nz, nkernel, nkernel);
    } else if (ndim == 3) {
        double temp1[nkernel_max] = {0.0};
        double temp2[nkernel_max] = {0.0};
        return apply_3d_stencil(u_ijk, stencil, stencil, stencil, temp1, temp2, 0, 1, 2, ny, nz, nkernel, nkernel, nkernel);
    } else {
        throw std::runtime_error("Invalid number of dimensions");
    }
}

double interpolate_cell_center(
    const double* u_ijk,
    const int p,
    const int nx,
    const int ny,
    const int nz
) {
    return _interpolate_cell_center_or_average(u_ijk, p, nx, ny, nz, true);
}

double interpolate_cell_average(
    const double* u_ijk,
    const int p,
    const int nx,
    const int ny,
    const int nz
) {
    return _interpolate_cell_center_or_average(u_ijk, p, nx, ny, nz, false);
}

void update_fv_fluxes(
    const py::array_t<double> _u_,
    py::array_t<double> _w_,
    py::array_t<double> _ucc_,
    const int p,
    const int nghost
) {
    if (_u_.ndim() != 4) {
        throw std::invalid_argument("_u_ must be a 4D array");
    }
    if (_ucc_.ndim() != 4) {
        throw std::invalid_argument("_ucc_ must be a 4D array");
    }
    for (py::ssize_t d = 0; d < _u_.ndim(); ++d) {
        if (_u_.shape(d) != _ucc_.shape(d)) {
            throw std::invalid_argument("_u_ and _ucc_ must have the same shape");
        }
    }
    const int nvars = static_cast<int>(_u_.shape(0));
    const int nx    = static_cast<int>(_u_.shape(1));
    const int ny    = static_cast<int>(_u_.shape(2));
    const int nz    = static_cast<int>(_u_.shape(3));
    const int ndim = count_ndim(nx, ny, nz);

    int xIdx1 = nghost;
    int xIdx2 = nx - nghost;
    int yIdx1 = ndim > 1 ? nghost : 0;
    int yIdx2 = ndim > 1 ? ny - nghost : 1;
    int zIdx1 = ndim > 2 ? nghost : 0;
    int zIdx2 = ndim > 2 ? nz - nghost : 1;

    for (int v = 0; v < nvars; ++v) {
        for (int i = xIdx1; i < xIdx2; ++i) {
            for (int j = yIdx1; j < yIdx2; ++j) {
                for (int k = zIdx1; k < zIdx2; ++k) {
                    _ucc_.mutable_at(v, i, j, k) = interpolate_cell_center(_u_.data(v, i, j, k), p, nx, ny, nz);
                    _w_.mutable_at(v, i, j, k) = interpolate_cell_average(_u_.data(v, i, j, k), p, nx, ny, nz);
                }
            }
        }
    }
}

PYBIND11_MODULE(_finite_volume_driver, m) {
    m.def("update_fv_fluxes", &update_fv_fluxes);
}
