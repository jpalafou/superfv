#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
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
    const bool left,
    const int p,
    const int axis,
    const int nx,
    const int ny,
    const int nz
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

double integrate_transverse_faces(
    const double* u_ijk,
    const int p,
    const int axis,
    const int ny,
    const int nz
) {
    // u_ijk points to u[..., i, j, k], with u having shape (..., nx, ny, nz)

    const int nkernel_max = 7;

    double stencil[nkernel_max] = {0.0};
    int nkernel = write_weights_for_transverse_integration_of_cell_average(stencil, nkernel_max, p);

    return apply_1d_stencil(u_ijk, stencil, axis, ny, nz, nkernel);
}

double integrate_transverse_nodes(const double* nodes, const int p, const int nnodes) {
    const int nnodes_max = 7;

    double stencil[nnodes_max] = {0.0};
    int nkernel = write_weights_for_transverse_integration_of_cell_average(stencil, nnodes_max, p);

    if (nnodes > nnodes_max) {
        throw std::invalid_argument("Number of nodes exceeds maximum supported");
    }
    if (nnodes != nkernel) {
        throw std::invalid_argument("Number of nodes does not match stencil size");
    }

    double out = 0.0;
    for (int i = 0; i < nkernel; ++i) {
        out += stencil[i] * nodes[i];
    }
    return out;
}

void update_fv_fluxes(
    const py::array_t<double> _u_,
    py::array_t<double> F,
    py::array_t<double> G,
    py::array_t<double> H,
    const int p,
    const int nghost,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    if (_u_.ndim() != 4) {
        throw std::invalid_argument("_u_ must be a 4D array");
    }
    if (F.ndim() != 4 || G.ndim() != 4 || H.ndim() != 4) {
        throw std::invalid_argument("F, G, and H must be 4D arrays");
    }

    const int nvars = static_cast<int>(_u_.shape(0));
    const int nx    = static_cast<int>(_u_.shape(1));
    const int ny    = static_cast<int>(_u_.shape(2));
    const int nz    = static_cast<int>(_u_.shape(3));
    const int ndim = count_ndim(nx, ny, nz);
    const int nnodes = (p + 1) / 2;
    const int nnodes_max = 7;

    int xIdxMax = nx - 2 * nghost + 1;
    int yIdxMax = ndim > 1 ? ny - 2 * nghost + 1 : 1;
    int zIdxMax = ndim > 2 ? nz - 2 * nghost + 1 : 1;

    for (int i = 0; i < xIdxMax; ++i) {
        for (int j = 0; j < yIdxMax; ++j) {
            for (int k = 0; k < zIdxMax; ++k) {
                if (ndim == 1) {
                    Conservatives left_cons{0, 0, 0, 0, 0, nullptr, 0};
                    Conservatives right_cons{0, 0, 0, 0, 0, nullptr, 0};
                    Conservatives flux{0, 0, 0, 0, 0, nullptr, 0};

                    Primitives left_prim{0, 0, 0, 0, 0, nullptr, 0};
                    Primitives right_prim{0, 0, 0, 0, 0, nullptr, 0};

                    // Interpolate node to the left of the interface
                    left_cons.rho = interpolate_face_center(_u_.data(0, i + nghost - 1, j, k), false, p, 0, nx, ny, nz);
                    left_cons.mx  = interpolate_face_center(_u_.data(1, i + nghost - 1, j, k), false, p, 0, nx, ny, nz);
                    left_cons.my  = interpolate_face_center(_u_.data(2, i + nghost - 1, j, k), false, p, 0, nx, ny, nz);
                    left_cons.mz  = interpolate_face_center(_u_.data(3, i + nghost - 1, j, k), false, p, 0, nx, ny, nz);
                    left_cons.E   = interpolate_face_center(_u_.data(4, i + nghost - 1, j, k), false, p, 0, nx, ny, nz);
                    cons_to_prim(left_cons, left_prim, gamma, isothermal, iso_cs);

                    // Interpolate node to the right of the interface
                    right_cons.rho = interpolate_face_center(_u_.data(0, i + nghost, j, k), true, p, 0, nx, ny, nz);
                    right_cons.mx  = interpolate_face_center(_u_.data(1, i + nghost, j, k), true, p, 0, nx, ny, nz);
                    right_cons.my  = interpolate_face_center(_u_.data(2, i + nghost, j, k), true, p, 0, nx, ny, nz);
                    right_cons.mz  = interpolate_face_center(_u_.data(3, i + nghost, j, k), true, p, 0, nx, ny, nz);
                    right_cons.E   = interpolate_face_center(_u_.data(4, i + nghost, j, k), true, p, 0, nx, ny, nz);
                    cons_to_prim(right_cons, right_prim, gamma, isothermal, iso_cs);

                    // Call Riemann solver
                    hllc_flux(left_prim, right_prim, flux, gamma, isothermal, iso_cs);

                    F.mutable_at(0, i, j, k) = flux.rho;
                    F.mutable_at(1, i, j, k) = flux.mx;
                    F.mutable_at(2, i, j, k) = flux.my;
                    F.mutable_at(3, i, j, k) = flux.mz;
                    F.mutable_at(4, i, j, k) = flux.E;
                } else {
                    throw std::runtime_error("Only 1D is implemented in update_fv_fluxes");
                }
            }
        }
    }
}

PYBIND11_MODULE(_finite_volume_driver, m) {
    m.def("update_fv_fluxes", &update_fv_fluxes);
}
