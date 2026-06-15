#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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
    double wl4, wl3, wl2, wl1, wcc, wr1, wr2, wr3, wr4;
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
}

std::ptrdiff_t get_stride(int axis, int ny, int nz) {
    // Returns the stride in the flattened array with dimensions (nvars, nx, ny, nz) for the given axis
    if (axis == 0) {
        return ny * nz;
    } else if (axis == 1) {
        return nz;
    } else if (axis == 2) {
        return 1;
    } else {
        throw std::invalid_argument("Invalid axis");
    }
}

double apply_1d_stencil(
    const double* u_ijk,
    const double* stencil,
    const int axis,
    const int ny,
    const int nz,
    const int nweights,
) {
    // u_ijk points to u[v, i, j, k], with u having shape (nvars, nx, ny, nz)
    // stencil has shape (nweights,)

    const std::ptrdiff_t stride = get_stride(axis, ny, nz);
    const int reach = (nweights - 1) / 2;
    
    double out = 0.0;
    for (int i = 0; i < nweights; ++i) {
        const int off = i - reach;
        out += stencil[i] * u_ijk[off * stride];
    }
    return out;
}

double apply_2d_stencil(
    const double* u_ijk,
    const double* stencil1,
    const double* stencil2,
    double* temp,
    const int axis1,
    const int axis2,
    const int ny,
    const int nz,
    const int nweights1,
    const int nweights2
) {
    // u_ijk points to u[v, i, j, k], with u having shape (nvars, nx, ny, nz)
    // stencil1 has shape (nweights1,)
    // stencil2 has shape (nweights2,)
    // temp has shape (nweights2,)

    const std::ptrdiff_t stride2 = get_stride(axis2, ny, nz);
    const int reach2 = (nweights2 - 1) / 2;

    // First apply stencil1 nweights2 separate times along axis1 to get temp
    for (int i = 0; i < nweights2; ++i) {
        const int off2 = i - reach2;
        temp[i] = apply_1d_stencil(u_ijk + off2 * stride2, stencil1, axis1, ny, nz, nweights1);
    }

    // Then apply stencil2 along axis2 to temp
    return apply_1d_stencil(temp + reach2, stencil2, 0, 1, 1, nweights2);
}

double apply_3d_stencil(
    const double* u_ijk,
    const double* stencil1,
    const double* stencil2,
    const double* stencil3,
    double* temp1,
    double* temp2,
    const int axis1,
    const int axis2,
    const int axis3,
    const int ny,
    const int nz,
    const int nweights1,
    const int nweights2,
    const int nweights3
) {
    // u_ijk points to u[v, i, j, k], with u having shape (nvars, nx, ny, nz)
    // stencil1 has shape (nweights1,)
    // stencil2 has shape (nweights2,)
    // stencil3 has shape (nweights3,)
    // temp1 has shape (nweights2,)
    // temp2 has shape (nweights3,)

    const std::ptrdiff_t stride3 = get_stride(axis3, ny, nz);
    const int reach3 = (nweights3 - 1) / 2;

    // First apply stencil1 and stencil2 to get temp1
    for (int i = 0; i < nweights3; ++i) {
        const int off3 = i - reach3;
        temp1[i] = apply_2d_stencil(u_ijk + off3 * stride3, stencil1, stencil2, temp2, axis1, axis2, ny, nz, nweights1, nweights2);
    }

    // Then apply stencil3 along axis3 to temp1
    return apply_1d_stencil(temp1 + reach3, stencil3, 0, 1, 1, nweights3);
}

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
