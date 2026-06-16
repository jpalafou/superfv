#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "stencil_application.h"

namespace py = pybind11;

double conservative_interpolation_LCR(
    double ul4,
    double ul3,
    double ul2,
    double ul1,
    double ucc,
    double ur1,
    double ur2,
    double ur3,
    double ur4,
    int pos,
    int p
) {
    double wl2 = 0.0;
    double wl1 = 0.0;
    double wcc = 0.0;
    double wr1 = 0.0;
    double wr2 = 0.0;

    if (pos < -1 || pos > 1) {
        throw std::invalid_argument("Invalid position for interpolation");
    }

    switch (p) {
        case 0:
            return ucc;
        case 1:
            switch (pos) {
                case -1:
                    wl1 = 1.0 / 4.0;
                    wcc = 1.0;
                    wr1 = -1.0 / 4.0;
                    break;
                case 0:
                    wl1 = 0.0;
                    wcc = 1.0;
                    wr1 = 0.0;
                    break;
                case 1:
                    wl1 = -1.0 / 4.0;
                    wcc = 1.0;
                    wr1 = 1.0 / 4.0;
                    break;
            }
            return wl1 * ul1 + wcc * ucc + wr1 * ur1;
        case 2:
            switch (pos) {
                case -1:
                    wl1 = 1.0 / 3.0;
                    wcc = 5.0 / 6.0;
                    wr1 = -1.0 / 6.0;
                    break;
                case 0:
                    wl1 = -1.0 / 24.0;
                    wcc = 13.0 / 12.0;
                    wr1 = -1.0 / 24.0;
                    break;
                case 1:
                    wl1 = -1.0 / 6.0;
                    wcc = 5.0 / 6.0;
                    wr1 = 1.0 / 3.0;
                    break;
            }
            return wl1 * ul1 + wcc * ucc + wr1 * ur1;
        case 3:
            switch (pos) {
                case -1:
                    wl2 = -1.0 / 24.0;
                    wl1 = 5.0 / 12.0;
                    wcc = 5.0 / 6.0;
                    wr1 = -1.0 / 4.0;
                    wr2 = 1.0 / 24.0;
                    break;
                case 0:
                    wl1 = -1.0 / 24.0;
                    wcc = 13.0 / 12.0;
                    wr1 = -1.0 / 24.0;
                    break;
                case 1:
                    wl2 = 1.0 / 24.0;
                    wl1 = -1.0 / 4.0;
                    wcc = 5.0 / 6.0;
                    wr1 = 5.0 / 12.0;
                    wr2 = -1.0 / 24.0;
                    break;
            }
            return wl2 * ul2 + wl1 * ul1 + wcc * ucc + wr1 * ur1 + wr2 * ur2;
    }
    throw std::invalid_argument("Invalid order of interpolation");
}

void interpolate_cell_centers(
    const py::array_t<double> _u_,
    py::array_t<double> _ucc_,
    int p,
    int nghost
) {
    if (_u_.ndim() != _ucc_.ndim()) {
        throw std::invalid_argument("u and ucc must have the same number of dimensions");
    }
    for (py::ssize_t d = 0; d < _u_.ndim(); ++d) {
        if (_u_.shape(d) != _ucc_.shape(d)) {
            throw std::invalid_argument("u and ucc must have the same shape");
        }
    }
    const int nvars = static_cast<int>(_u_.shape(0));
    const int nx    = static_cast<int>(_u_.shape(1));
    const int ny    = static_cast<int>(_u_.shape(2));
    const int nz    = static_cast<int>(_u_.shape(3));
    const int nkernel_max = 7;

    double stencil[nkernel_max] = {0.0};
    int nkernel = 0;
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

    int ndim = 0;
    if (nx > 1) {
        if (ny > 1) {
            if (nz > 1) {
                ndim = 3;
            } else {
                ndim = 2;
            }
        } else {
            ndim = 1;
        }
    }

    if (ndim == 1) {
        for (int v = 0; v < nvars; ++v) {
            for (int i = nghost; i < nx - nghost; ++i) {
                _ucc_.mutable_at(v, i, 0, 0) = apply_1d_stencil(_u_.data(v, i, 0, 0), stencil, 0, ny, nz, nkernel);
            }
        }
    } else if (ndim == 2) {
        double temp[nkernel_max] = {0.0};

        for (int v = 0; v < nvars; ++v) {
            for (int i = nghost; i < nx - nghost; ++i) {
                for (int j = nghost; j < ny - nghost; ++j) {
                    _ucc_.mutable_at(v, i, j, 0) = apply_2d_stencil(_u_.data(v, i, j, 0), stencil, stencil, temp, 0, 1, ny, nz, nkernel, nkernel);
                }
            }
        }
    } else if (ndim == 3) {
        double temp1[nkernel_max] = {0.0};
        double temp2[nkernel_max] = {0.0};

        for (int v = 0; v < nvars; ++v) {
            for (int i = nghost; i < nx - nghost; ++i) {
                for (int j = nghost; j < ny - nghost; ++j) {
                    for (int k = nghost; k < nz - nghost; ++k) {
                        _ucc_.mutable_at(v, i, j, k) = apply_3d_stencil(_u_.data(v, i, j, k), stencil, stencil, stencil, temp1, temp2, 0, 1, 2, ny, nz, nkernel, nkernel, nkernel);
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("Invalid number of dimensions");
    }
}

void update_fv_fluxes(
    py::array _F_,
    py::array _G_,
    py::array _H_,
    int nvars,
    int nx,
    int ny,
    int nz,
    int nghost,
    bool x_active,
    bool y_active,
    bool z_active
) {
    int _nx_ = x_active ? nx + 2 * nghost : 1;
    int _ny_ = y_active ? ny + 2 * nghost : 1;
    int _nz_ = z_active ? nz + 2 * nghost : 1;
    int x_inner_i0 = x_active ? nghost : 0;
    int x_inner_i1 = x_active ? nx + nghost : 1;
    int y_inner_j0 = y_active ? nghost : 0;
    int y_inner_j1 = y_active ? ny + nghost : 1;
    int z_inner_k0 = z_active ? nghost : 0;
    int z_inner_k1 = z_active ? nz + nghost : 1;

    for (int i = x_inner_i0; i < x_inner_i1; ++i) {
        for (int j = y_inner_j0; j < y_inner_j1; ++j) {
            for (int k = z_inner_k0; k < z_inner_k1; ++k) {
            }
        }
    }

    if (x_active) {
        auto _f_ = _F_.mutable_unchecked<double, 4>();
        for (py::ssize_t v = 0; v < nvars; ++v) {
            for (int i = 0; i < nx + 1; ++i) {
                for (int j = 0; j < _ny_; ++j) {
                    for (int k = 0; k < _nz_; ++k) {
                        _f_(v, i, j, k) = 0.0;
                    }
                }
            }
        }
    }

    if (y_active) {
        auto _g_ = _G_.mutable_unchecked<double, 4>();
        for (py::ssize_t v = 0; v < nvars; ++v) {
            for (int i = 0; i < _nx_; ++i) {
                for (int j = 0; j < ny + 1; ++j) {
                    for (int k = 0; k < _nz_; ++k) {
                        _g_(v, i, j, k) = 0.0;
                    }
                }
            }
        }
    }

    if (z_active) {
        auto _h_ = _H_.mutable_unchecked<double, 4>();
        for (py::ssize_t v = 0; v < nvars; ++v) {
            for (int i = 0; i < _nx_; ++i) {
                for (int j = 0; j < _ny_; ++j) {
                    for (int k = 0; k < nz + 1; ++k) {
                        _h_(v, i, j, k) = 0.0;
                    }
                }
            }
        }
    }
}

PYBIND11_MODULE(_finite_volume_driver, m) {
    m.def("update_fv_fluxes", &update_fv_fluxes);
    m.def("interpolate_cell_centers", &interpolate_cell_centers);
}
