#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "constants.h"
#include "stencils.h"
#include "stencil_application.h"
#include "hydro.h"
#include "riemann_solvers.h"

namespace py = pybind11;

int count_ndim(const int nx, const int ny, const int nz) {
    if (!(nx >= ny && ny >= nz)) {
        throw std::invalid_argument("Invalid grid dimensions");
    }
    return static_cast<int>(nx > 1) + static_cast<int>(ny > 1) + static_cast<int>(nz > 1);
}

double interpolate_cell_center_or_average(
    const double* u_ijk,
    const Stencil& stencil,
    const int nx,
    const int ny,
    const int nz
) {
    // u_ijk points to u[..., i, j, k], with u having shape (..., nx, ny, nz)

    const int ndim = count_ndim(nx, ny, nz);

    if (ndim == 1) {
        return apply_1d_stencil(u_ijk, stencil, 0, ny, nz);
    } else if (ndim == 2) {
        double temp[MAX_NODES] = {0.0};
        return apply_2d_stencil(u_ijk, stencil, stencil, temp, 0, 1, ny, nz);
    } else if (ndim == 3) {
        double temp1[MAX_NODES] = {0.0};
        double temp2[MAX_NODES] = {0.0};
        return apply_3d_stencil(u_ijk, stencil, stencil, stencil, temp1, temp2, 0, 1, 2, ny, nz);
    } else {
        throw std::runtime_error("Invalid number of dimensions");
    }
}

double interpolate_face_center(
    const double* u_ijk,
    const Stencil& lr_stencil,
    const Stencil& c_stencil,
    const int axis,
    const int nx,
    const int ny,
    const int nz
) {
    // u_ijk points to u[..., i, j, k], with u having shape (..., nx, ny, nz)

    const int ndim = count_ndim(nx, ny, nz);

    if (ndim == 1) {
        return apply_1d_stencil(u_ijk, lr_stencil, axis, ny, nz);
    } else if (ndim == 2) {
        double temp[MAX_NODES] = {0.0};
        return apply_2d_stencil(u_ijk, lr_stencil, c_stencil, temp, axis, (axis + 1) % 2, ny, nz);
    } else if (ndim == 3) {
        double temp1[MAX_NODES] = {0.0};
        double temp2[MAX_NODES] = {0.0};
        return apply_3d_stencil(u_ijk, lr_stencil, c_stencil, c_stencil, temp1, temp2, axis, (axis + 1) % 3, (axis + 2) % 3, ny, nz);
    } else {
        throw std::runtime_error("Invalid number of dimensions");
    }
}

double integrate_transverse_faces(
    const double* u_ijk,
    const Stencil& stencil,
    const int axis,
    const int nx,
    const int ny,
    const int nz
) {
    // u_ijk points to u[..., i, j, k], with u having shape (..., nx, ny, nz)

    const int ndim = count_ndim(nx, ny, nz);

    if (ndim == 1) {
        throw std::runtime_error("Cannot integrate transverse faces in 1D");
    } else if (ndim == 2) {
        return apply_1d_stencil(u_ijk, stencil, (axis + 1) % 2, ny, nz);
    } else if (ndim == 3) {
        double temp[MAX_NODES] = {0.0};
        return apply_2d_stencil(u_ijk, stencil, stencil, temp, (axis + 1) % 3, (axis + 2) % 3, ny, nz);
    } else {
        throw std::runtime_error("Invalid number of dimensions");
    }
}

void update_1D_fv_fluxes(
    const py::array_t<double> _u_,
    py::array_t<double> F,
    const int p,
    const int nghost,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    const int nx = static_cast<int>(_u_.shape(1));
    const Stencil l_stencil = conservative_interpolation_of_left_or_right_face(p, true);
    const Stencil r_stencil = conservative_interpolation_of_left_or_right_face(p, false);
    const Stencil c_stencil = conservative_interpolation_of_cell_center(p);

    for (int i = 0; i < nx - 2 * nghost + 1; ++i) {
        Conservatives left_cons{}, right_cons{}, flux{};
        Primitives left_prim{}, right_prim{};
        int io;

        // Interpolate node to the left of the interface
        io = i + nghost - 1;
        left_cons.rho = interpolate_face_center(_u_.data(0, io, 0, 0), r_stencil, c_stencil, 0, nx, 1, 1);
        left_cons.mx  = interpolate_face_center(_u_.data(1, io, 0, 0), r_stencil, c_stencil, 0, nx, 1, 1);
        left_cons.my  = interpolate_face_center(_u_.data(2, io, 0, 0), r_stencil, c_stencil, 0, nx, 1, 1);
        left_cons.mz  = interpolate_face_center(_u_.data(3, io, 0, 0), r_stencil, c_stencil, 0, nx, 1, 1);
        left_cons.E   = interpolate_face_center(_u_.data(4, io, 0, 0), r_stencil, c_stencil, 0, nx, 1, 1);
        cons_to_prim(left_cons, left_prim, gamma, isothermal, iso_cs);

        // Interpolate node to the right of the interface
        io = i + nghost;
        right_cons.rho = interpolate_face_center(_u_.data(0, io, 0, 0), l_stencil, c_stencil, 0, nx, 1, 1);
        right_cons.mx  = interpolate_face_center(_u_.data(1, io, 0, 0), l_stencil, c_stencil, 0, nx, 1, 1);
        right_cons.my  = interpolate_face_center(_u_.data(2, io, 0, 0), l_stencil, c_stencil, 0, nx, 1, 1);
        right_cons.mz  = interpolate_face_center(_u_.data(3, io, 0, 0), l_stencil, c_stencil, 0, nx, 1, 1);
        right_cons.E   = interpolate_face_center(_u_.data(4, io, 0, 0), l_stencil, c_stencil, 0, nx, 1, 1);
        cons_to_prim(right_cons, right_prim, gamma, isothermal, iso_cs);

        // Call Riemann solver
        hllc_flux(left_prim, right_prim, flux, 1, gamma, isothermal, iso_cs);
        F.mutable_at(0, i, 0, 0) = flux.rho;
        F.mutable_at(1, i, 0, 0) = flux.mx;
        F.mutable_at(2, i, 0, 0) = flux.my;
        F.mutable_at(3, i, 0, 0) = flux.mz;
        F.mutable_at(4, i, 0, 0) = flux.E;
    }
}

void update_2D_fv_fluxes(
    const py::array_t<double> _u_,
    py::array_t<double> F,
    py::array_t<double> G,
    py::array_t<double> _f_node_buffer_,
    const int p,
    const int nghost,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    const int nvars = static_cast<int>(_u_.shape(0));
    const int nx    = static_cast<int>(_u_.shape(1));
    const int ny    = static_cast<int>(_u_.shape(2));
    const Stencil l_stencil = conservative_interpolation_of_left_or_right_face(p, true);
    const Stencil r_stencil = conservative_interpolation_of_left_or_right_face(p, false);
    const Stencil c_stencil = conservative_interpolation_of_cell_center(p);
    const Stencil t_stencil = transverse_integration_of_cell_average(p);
    const int c_reach = (l_stencil.n - 1) / 2;

    for (int axis = 0; axis < 2; ++axis) {
        int ilb = c_reach, iub = nx - c_reach;
        int jlb = c_reach, jub = ny - c_reach;

        if (axis == 0) {
            ilb = nghost; iub = nx - nghost + 1;
        } else {
            jlb = nghost; jub = ny - nghost + 1;
        }
        for (int i = ilb; i < iub; ++i) {
            for (int j = jlb; j < jub; ++j) {
                // Compute flux node at each interface
                Conservatives left_cons{}, right_cons{}, flux{};
                Primitives left_prim{}, right_prim{};

                // Interpolate node to the left of the interface
                int io = axis == 0 ? i - 1: i;
                int jo = axis == 1 ? j - 1: j;
                left_cons.rho = interpolate_face_center(_u_.data(0, io, jo, 0), r_stencil, c_stencil, axis, nx, ny, 1);
                left_cons.mx  = interpolate_face_center(_u_.data(1, io, jo, 0), r_stencil, c_stencil, axis, nx, ny, 1);
                left_cons.my  = interpolate_face_center(_u_.data(2, io, jo, 0), r_stencil, c_stencil, axis, nx, ny, 1);
                left_cons.mz  = interpolate_face_center(_u_.data(3, io, jo, 0), r_stencil, c_stencil, axis, nx, ny, 1);
                left_cons.E   = interpolate_face_center(_u_.data(4, io, jo, 0), r_stencil, c_stencil, axis, nx, ny, 1);
                cons_to_prim(left_cons, left_prim, gamma, isothermal, iso_cs);

                // Interpolate node to the right of the interface
                right_cons.rho = interpolate_face_center(_u_.data(0, i, j, 0), l_stencil, c_stencil, axis, nx, ny, 1);
                right_cons.mx  = interpolate_face_center(_u_.data(1, i, j, 0), l_stencil, c_stencil, axis, nx, ny, 1);
                right_cons.my  = interpolate_face_center(_u_.data(2, i, j, 0), l_stencil, c_stencil, axis, nx, ny, 1);
                right_cons.mz  = interpolate_face_center(_u_.data(3, i, j, 0), l_stencil, c_stencil, axis, nx, ny, 1);
                right_cons.E   = interpolate_face_center(_u_.data(4, i, j, 0), l_stencil, c_stencil, axis, nx, ny, 1);
                cons_to_prim(right_cons, right_prim, gamma, isothermal, iso_cs);

                // Call Riemann solver and store nodal fluxes
                hllc_flux(left_prim, right_prim, flux, axis + 1, gamma, isothermal, iso_cs);
                _f_node_buffer_.mutable_at(0, i, j, 0) = flux.rho;
                _f_node_buffer_.mutable_at(1, i, j, 0) = flux.mx;
                _f_node_buffer_.mutable_at(2, i, j, 0) = flux.my;
                _f_node_buffer_.mutable_at(3, i, j, 0) = flux.mz;
                _f_node_buffer_.mutable_at(4, i, j, 0) = flux.E;
            }
        }

        // Traverse each cell non-ghost cell again to compute transverse integral
        iub = nx - 2 * nghost;
        jub = ny - 2 * nghost;
        if (axis == 0) {iub += 1;} else {jub += 1;}
        for (int i = 0; i < iub; ++i) {
            for (int j = 0; j < jub; ++j) {
                if (axis == 0) {
                    for (int v = 0; v < nvars; ++v) {
                        F.mutable_at(v, i, j, 0) = apply_1d_stencil(_f_node_buffer_.data(v, i + nghost, j + nghost, 0), t_stencil, 1, ny, 1);
                    }
                } else {
                    for (int v = 0; v < nvars; ++v) {
                        G.mutable_at(v, i, j, 0) = apply_1d_stencil(_f_node_buffer_.data(v, i + nghost, j + nghost, 0), t_stencil, 0, ny, 1);
                    }
                }
            }
        }
    }
}

void update_3D_fv_fluxes(
    const py::array_t<double> _u_,
    py::array_t<double> F,
    py::array_t<double> G,
    py::array_t<double> H,
    py::array_t<double> _f_node_buffer_,
    const int p,
    const int nghost,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    const int nvars = static_cast<int>(_u_.shape(0));
    const int nx    = static_cast<int>(_u_.shape(1));
    const int ny    = static_cast<int>(_u_.shape(2));
    const int nz    = static_cast<int>(_u_.shape(3));
    const Stencil l_stencil = conservative_interpolation_of_left_or_right_face(p, true);
    const Stencil r_stencil = conservative_interpolation_of_left_or_right_face(p, false);
    const Stencil c_stencil = conservative_interpolation_of_cell_center(p);
    const Stencil t_stencil = transverse_integration_of_cell_average(p);
    const int c_reach = (l_stencil.n - 1) / 2;

    for (int axis = 0; axis < 3; ++axis) {
        int ilb = c_reach, iub = nx - c_reach;
        int jlb = c_reach, jub = ny - c_reach;
        int klb = c_reach, kub = nz - c_reach;

        if (axis == 0) {
            ilb = nghost; iub = nx - nghost + 1;
        } else if (axis == 1) {
            jlb = nghost; jub = ny - nghost + 1;
        } else {
            klb = nghost; kub = nz - nghost + 1;
        }
        for (int i = ilb; i < iub; ++i) {
            for (int j = jlb; j < jub; ++j) {
                for (int k = klb; k < kub; ++k) {
                    // Compute flux node at each interface
                    Conservatives left_cons{}, right_cons{}, flux{};
                    Primitives left_prim{}, right_prim{};

                    // Interpolate node to the left of the interface
                    int io = axis == 0 ? i - 1: i;
                    int jo = axis == 1 ? j - 1: j;
                    int ko = axis == 2 ? k - 1: k;
                    left_cons.rho = interpolate_face_center(_u_.data(0, io, jo, ko), r_stencil, c_stencil, axis, nx, ny, nz);
                    left_cons.mx  = interpolate_face_center(_u_.data(1, io, jo, ko), r_stencil, c_stencil, axis, nx, ny, nz);
                    left_cons.my  = interpolate_face_center(_u_.data(2, io, jo, ko), r_stencil, c_stencil, axis, nx, ny, nz);
                    left_cons.mz  = interpolate_face_center(_u_.data(3, io, jo, ko), r_stencil, c_stencil, axis, nx, ny, nz);
                    left_cons.E   = interpolate_face_center(_u_.data(4, io, jo, ko), r_stencil, c_stencil, axis, nx, ny, nz);
                    cons_to_prim(left_cons, left_prim, gamma, isothermal, iso_cs);

                    // Interpolate node to the right of the interface
                    right_cons.rho = interpolate_face_center(_u_.data(0, i, j, k), l_stencil, c_stencil, axis, nx, ny, nz);
                    right_cons.mx  = interpolate_face_center(_u_.data(1, i, j, k), l_stencil, c_stencil, axis, nx, ny, nz);
                    right_cons.my  = interpolate_face_center(_u_.data(2, i, j, k), l_stencil, c_stencil, axis, nx, ny, nz);
                    right_cons.mz  = interpolate_face_center(_u_.data(3, i, j, k), l_stencil, c_stencil, axis, nx, ny, nz);
                    right_cons.E   = interpolate_face_center(_u_.data(4, i, j, k), l_stencil, c_stencil, axis, nx, ny, nz);
                    cons_to_prim(right_cons, right_prim, gamma, isothermal, iso_cs);

                    // Call Riemann solver and store nodal fluxes
                    hllc_flux(left_prim, right_prim, flux, axis + 1, gamma, isothermal, iso_cs);
                    _f_node_buffer_.mutable_at(0, i, j, k) = flux.rho;
                    _f_node_buffer_.mutable_at(1, i, j, k) = flux.mx;
                    _f_node_buffer_.mutable_at(2, i, j, k) = flux.my;
                    _f_node_buffer_.mutable_at(3, i, j, k) = flux.mz;
                    _f_node_buffer_.mutable_at(4, i, j, k) = flux.E;
                }
            }
        }

        // Traverse each cell non-ghost cell again to compute transverse integral
        iub = nx - 2 * nghost;
        jub = ny - 2 * nghost;
        kub = nz - 2 * nghost;
        if (axis == 0) {iub += 1;} else if (axis == 1) {jub += 1;} else {kub += 1;}
        for (int i = 0; i < iub; ++i) {
            for (int j = 0; j < jub; ++j) {
                for (int k = 0; k < kub; ++k) {
                    double temp[MAX_NODES] = {};
                    if (axis == 0) {
                        for (int v = 0; v < nvars; ++v) {
                            F.mutable_at(v, i, j, k) = apply_2d_stencil(_f_node_buffer_.data(v, i + nghost, j + nghost, k + nghost), t_stencil, t_stencil, temp, 1, 2, ny, nz);
                        }
                    } else if (axis == 1) {
                        for (int v = 0; v < nvars; ++v) {
                            G.mutable_at(v, i, j, k) = apply_2d_stencil(_f_node_buffer_.data(v, i + nghost, j + nghost, k + nghost), t_stencil, t_stencil, temp, 0, 2, ny, nz);
                        }
                    } else {
                        for (int v = 0; v < nvars; ++v) {
                            H.mutable_at(v, i, j, k) = apply_2d_stencil(_f_node_buffer_.data(v, i + nghost, j + nghost, k + nghost), t_stencil, t_stencil, temp, 0, 1, ny, nz);
                        }
                    }
                }
            }
        }
    }
}

void require_contiguous_shape(
    const py::array_t<double>& arr,
    std::string name,
    const int nvars,
    const int nx,
    const int ny,
    const int nz
) {
    if (arr.ndim() != 4) {
        throw std::invalid_argument(name + " must be a 4D array");
    }
    if (arr.shape(0) != nvars || arr.shape(1) != nx || arr.shape(2) != ny || arr.shape(3) != nz) {
        throw std::invalid_argument(name + " must have shape (" + std::to_string(nvars) + ", " + std::to_string(nx) + ", " + std::to_string(ny) + ", " + std::to_string(nz) + ")");
    }
    if (!(arr.flags() & py::array::c_style)) {
        throw std::invalid_argument(name + " must be C-contiguous");
    }
}

void update_fv_fluxes(
    const py::array_t<double> _u_,
    py::array_t<double> F,
    py::array_t<double> G,
    py::array_t<double> H,
    py::array_t<double> _f_node_buffer_,
    const int p,
    const int nghost,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    if (_u_.ndim() != 4) {
        throw std::invalid_argument("_u_ must be a 4D array");
    }

    const int nvars = static_cast<int>(_u_.shape(0));
    const int _nx_ = static_cast<int>(_u_.shape(1));
    const int _ny_ = static_cast<int>(_u_.shape(2));
    const int _nz_ = static_cast<int>(_u_.shape(3));
    const int ndim = count_ndim(_nx_, _ny_, _nz_);
    const int nx = _nx_ - 2 * nghost;
    const int ny = _ny_ - 2 * nghost;
    const int nz = _nz_ - 2 * nghost;

    require_contiguous_shape(_u_, "_u_", nvars, _nx_, _ny_, _nz_);
    require_contiguous_shape(F, "F", nvars, nx + 1, ny, nz);
    require_contiguous_shape(G, "G", nvars, nx, ny + 1, nz);
    require_contiguous_shape(H, "H", nvars, nx, ny, nz + 1);
    require_contiguous_shape(_f_node_buffer_, "_f_node_buffer_", nvars, _nx_, _ny_, _nz_);

    if (ndim == 1) {
        update_1D_fv_fluxes(_u_, F, p, nghost, gamma, isothermal, iso_cs);
    } else if (ndim == 2) {
        update_2D_fv_fluxes(_u_, F, G, _f_node_buffer_, p, nghost, gamma, isothermal, iso_cs);
    } else if (ndim == 3) {
        update_3D_fv_fluxes(_u_, F, G, H, _f_node_buffer_, p, nghost, gamma, isothermal, iso_cs);
    } else {
        throw std::runtime_error("Invalid number of dimensions");
    }
}

PYBIND11_MODULE(_finite_volume_driver, m) {
    m.def("update_fv_fluxes", &update_fv_fluxes);
}
