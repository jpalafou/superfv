#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void update_fv_fluxes_cpp(
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
    m.def("update_fv_fluxes_cpp", &update_fv_fluxes_cpp);
}
