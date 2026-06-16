#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "stencils.h"
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
        nkernel = write_weights_for_conservative_interpolation_of_cell_center(stencil, nkernel_max, p);
    } else {
        nkernel = write_weights_for_transverse_integration_of_cell_average(stencil, nkernel_max, p);
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

double interpolate_face_center(
    const double* u_ijk,
    const int axis,
    const bool left,
    const int p,
    const int nx,
    const int ny,
    const int nz,
    const bool cell_centers
) {
    // u_ijk points to u[..., i, j, k], with u having shape (..., nx, ny, nz)

    const int ndim = count_ndim(nx, ny, nz);
    const int nkernel_max = 9;

    double LRstencil[nkernel_max] = {0.0};
    double Cstencil[nkernel_max] = {0.0};
    int LRnkernel = 0;
    int Cnkernel = 0;
    LRnkernel = write_weights_for_conservative_interpolation_of_left_or_right_face(LRstencil, nkernel_max, p, left);
    Cnkernel = write_weights_for_conservative_interpolation_of_cell_center(Cstencil, nkernel_max, p);

    if (ndim == 1) {
        return apply_1d_stencil(u_ijk, LRstencil, axis, ny, nz, LRnkernel);
    } else if (ndim == 2) {
        double temp[nkernel_max] = {0.0};
        return apply_2d_stencil(u_ijk, LRstencil, Cstencil, temp, axis, (axis + 1) % 2, ny, nz, LRnkernel, Cnkernel);
    } else if (ndim == 3) {
        double temp1[nkernel_max] = {0.0};
        double temp2[nkernel_max] = {0.0};
        return apply_3d_stencil(u_ijk, LRstencil, Cstencil, Cstencil, temp1, temp2, axis, (axis + 1) % 2, (axis + 2) % 2, ny, nz, LRnkernel, Cnkernel, Cnkernel);
    } else {
        throw std::runtime_error("Invalid number of dimensions");
    }
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
