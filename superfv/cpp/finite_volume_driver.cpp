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
        return apply_3d_stencil(u_ijk, LRstencil, Cstencil, Cstencil, temp1, temp2, axis, (axis + 1) % 3, (axis + 2) % 3, ny, nz, LRnkernel, Cnkernel, Cnkernel);
    } else {
        throw std::runtime_error("Invalid number of dimensions");
    }
}

double integrate_transverse_faces(
    const double* u_ijk,
    const int p,
    const int axis,
    const int nx,
    const int ny,
    const int nz
) {
    // u_ijk points to u[..., i, j, k], with u having shape (..., nx, ny, nz)

    const int ndim = count_ndim(nx, ny, nz);
    const int nkernel_max = 7;

    double stencil[nkernel_max] = {0.0};
    int nkernel = write_weights_for_transverse_integration_of_cell_average(stencil, nkernel_max, p);

    if (ndim == 1) {
        throw std::runtime_error("Cannot integrate transverse faces in 1D");
    } else if (ndim == 2) {
        return apply_1d_stencil(u_ijk, stencil, (axis + 1) % 2, ny, nz, nkernel);
    } else if (ndim == 3) {
        double temp[nkernel_max] = {0.0};
        return apply_2d_stencil(u_ijk, stencil, stencil, temp, (axis + 1) % 3, (axis + 2) % 3, ny, nz, nkernel, nkernel);
    } else {
        throw std::runtime_error("Invalid number of dimensions");
    }
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

void update_1D_fv_fluxes(
    const py::array_t<double> _u_,
    py::array_t<double> F,
    const int p,
    const int nghost,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    if (_u_.ndim() != 4) {
        throw std::invalid_argument("_u_ must be a 4D array");
    }
    if (F.ndim() != 4) {
        throw std::invalid_argument("F must be a 4D array");
    }

    const int nx = static_cast<int>(_u_.shape(1));

    for (int i = 0; i < nx - 2 * nghost + 1; ++i) {
        Conservatives left_cons{}, right_cons{}, flux{};
        Primitives left_prim{}, right_prim{};
        int io;

        // Interpolate node to the left of the interface
        io = i + nghost - 1;
        left_cons.rho = interpolate_face_center(_u_.data(0, io, 0, 0), false, p, 0, nx, 1, 1);
        left_cons.mx  = interpolate_face_center(_u_.data(1, io, 0, 0), false, p, 0, nx, 1, 1);
        left_cons.my  = interpolate_face_center(_u_.data(2, io, 0, 0), false, p, 0, nx, 1, 1);
        left_cons.mz  = interpolate_face_center(_u_.data(3, io, 0, 0), false, p, 0, nx, 1, 1);
        left_cons.E   = interpolate_face_center(_u_.data(4, io, 0, 0), false, p, 0, nx, 1, 1);
        cons_to_prim(left_cons, left_prim, gamma, isothermal, iso_cs);

        // Interpolate node to the right of the interface
        io = i + nghost;
        right_cons.rho = interpolate_face_center(_u_.data(0, io, 0, 0), true, p, 0, nx, 1, 1);
        right_cons.mx  = interpolate_face_center(_u_.data(1, io, 0, 0), true, p, 0, nx, 1, 1);
        right_cons.my  = interpolate_face_center(_u_.data(2, io, 0, 0), true, p, 0, nx, 1, 1);
        right_cons.mz  = interpolate_face_center(_u_.data(3, io, 0, 0), true, p, 0, nx, 1, 1);
        right_cons.E   = interpolate_face_center(_u_.data(4, io, 0, 0), true, p, 0, nx, 1, 1);
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
    const int p,
    const int nghost,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    if (_u_.ndim() != 4) {
        throw std::invalid_argument("_u_ must be a 4D array");
    }
    if (F.ndim() != 4) {
        throw std::invalid_argument("F must be a 4D array");
    }
    if (G.ndim() != 4) {
        throw std::invalid_argument("G must be a 4D array");
    }

    const int nvars = static_cast<int>(_u_.shape(0));
    const int nx    = static_cast<int>(_u_.shape(1));
    const int ny    = static_cast<int>(_u_.shape(2));
    const int nnodes_max = 7;
    const int nvars_max = 10;

    double stencil[nnodes_max] = {};
    int nnodes = write_weights_for_transverse_integration_of_cell_average(stencil, nnodes_max, p);
    int reach = (nnodes - 1) / 2;

    if (nnodes > nnodes_max) {
        throw std::invalid_argument("Polynomial order p is too high for the maximum supported number of nodes");
    }

    for (int i = 0; i < nx - 2 * nghost + 1; ++i) {
        for (int j = 0; j < ny - 2 * nghost + 1; ++j) {
            for (int axis = 0; axis < 2; ++axis) {
                // Skip corners
                if (axis == 0 && j == ny - 2 * nghost) {continue;}
                if (axis == 1 && i == nx - 2 * nghost) {continue;}

                // Collect nodes for transverse integration
                double nodal_fluxes[nvars_max * nnodes_max] = {};
                for (int q = 0; q < nnodes; ++q) {
                    const int offset = q - reach;

                    Conservatives left_cons{}, right_cons{}, flux{};
                    Primitives left_prim{}, right_prim{};
                    int io, jo;

                    // Interpolate node to the left of the interface
                    io = axis == 0 ? i + nghost - 1: i + nghost + offset;
                    jo = axis == 1 ? j + nghost - 1: j + nghost + offset;
                    left_cons.rho = interpolate_face_center(_u_.data(0, io, jo, 0), false, p, axis, nx, ny, 1);
                    left_cons.mx  = interpolate_face_center(_u_.data(1, io, jo, 0), false, p, axis, nx, ny, 1);
                    left_cons.my  = interpolate_face_center(_u_.data(2, io, jo, 0), false, p, axis, nx, ny, 1);
                    left_cons.mz  = interpolate_face_center(_u_.data(3, io, jo, 0), false, p, axis, nx, ny, 1);
                    left_cons.E   = interpolate_face_center(_u_.data(4, io, jo, 0), false, p, axis, nx, ny, 1);
                    cons_to_prim(left_cons, left_prim, gamma, isothermal, iso_cs);

                    // Interpolate node to the right of the interface
                    io = axis == 0 ? i + nghost: i + nghost + offset;
                    jo = axis == 1 ? j + nghost: j + nghost + offset;
                    right_cons.rho = interpolate_face_center(_u_.data(0, io, jo, 0), true, p, axis, nx, ny, 1);
                    right_cons.mx  = interpolate_face_center(_u_.data(1, io, jo, 0), true, p, axis, nx, ny, 1);
                    right_cons.my  = interpolate_face_center(_u_.data(2, io, jo, 0), true, p, axis, nx, ny, 1);
                    right_cons.mz  = interpolate_face_center(_u_.data(3, io, jo, 0), true, p, axis, nx, ny, 1);
                    right_cons.E   = interpolate_face_center(_u_.data(4, io, jo, 0), true, p, axis, nx, ny, 1);
                    cons_to_prim(right_cons, right_prim, gamma, isothermal, iso_cs);

                    // Call Riemann solver and store nodal fluxes
                    hllc_flux(left_prim, right_prim, flux, axis + 1, gamma, isothermal, iso_cs);
                    nodal_fluxes[0 * nnodes_max + q] = flux.rho;
                    nodal_fluxes[1 * nnodes_max + q] = flux.mx;
                    nodal_fluxes[2 * nnodes_max + q] = flux.my;
                    nodal_fluxes[3 * nnodes_max + q] = flux.mz;
                    nodal_fluxes[4 * nnodes_max + q] = flux.E;
                }

                // Integrate nodal fluxes to get face-centered flux
                ptrdiff_t center_offset = nnodes / 2;
                if (axis == 0) {
                    for (int v = 0; v < nvars; ++v) {
                        F.mutable_at(v, i, j, 0) = apply_1d_stencil(&nodal_fluxes[v * nnodes_max + center_offset], stencil, 0, 1, 1, nnodes);
                    }
                } else {
                    for (int v = 0; v < nvars; ++v) {
                        G.mutable_at(v, i, j, 0) = apply_1d_stencil(&nodal_fluxes[v * nnodes_max + center_offset], stencil, 0, 1, 1, nnodes);
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
    const int p,
    const int nghost,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    if (_u_.ndim() != 4) {
        throw std::invalid_argument("_u_ must be a 4D array");
    }
    if (F.ndim() != 4) {
        throw std::invalid_argument("F must be a 4D array");
    }
    if (G.ndim() != 4) {
        throw std::invalid_argument("G must be a 4D array");
    }
    if (H.ndim() != 4) {
        throw std::invalid_argument("H must be a 4D array");
    }

    const int nvars = static_cast<int>(_u_.shape(0));
    const int nx    = static_cast<int>(_u_.shape(1));
    const int ny    = static_cast<int>(_u_.shape(2));
    const int nz    = static_cast<int>(_u_.shape(3));
    const int nnodes_max = 7;
    const int nvars_max = 10;

    double stencil[nnodes_max] = {};
    int nnodes = write_weights_for_transverse_integration_of_cell_average(stencil, nnodes_max, p);
    int reach = (nnodes - 1) / 2;

    if (nnodes > nnodes_max) {
        throw std::invalid_argument("Polynomial order p is too high for the maximum supported number of nodes");
    }

    for (int i = 0; i < nx - 2 * nghost + 1; ++i) {
        for (int j = 0; j < ny - 2 * nghost + 1; ++j) {
            for (int k = 0; k < nz - 2 * nghost + 1; ++k) {
                for (int axis = 0; axis < 3; ++axis) {
                    // Skip corners
                    if (axis == 0 && ((j == ny - 2 * nghost) || (k == nz - 2 * nghost))) {continue;}
                    if (axis == 1 && ((i == nx - 2 * nghost) || (k == nz - 2 * nghost))) {continue;}
                    if (axis == 2 && ((i == nx - 2 * nghost) || (j == ny - 2 * nghost))) {continue;}

                    // Collect nodes for transverse integration
                    double nodal_fluxes[nvars_max * nnodes_max * nnodes_max] = {};
                    double temp[nvars_max * nnodes_max] = {};
                    for (int q1 = 0; q1 < nnodes; ++q1) {
                        for (int q2 = 0; q2 < nnodes; ++q2) {
                            const int offset1 = q1 - reach;
                            const int offset2 = q2 - reach;

                            Conservatives left_cons{}, right_cons{}, flux{};
                            Primitives left_prim{}, right_prim{};
                            int io, jo, ko;

                            // Interpolate node to the left of the interface
                            if (axis == 0) {
                                io = i + nghost - 1;
                                jo = j + nghost + offset1;
                                ko = k + nghost + offset2;
                            } else if (axis == 1) {
                                io = i + nghost + offset1;
                                jo = j + nghost - 1;
                                ko = k + nghost + offset2;
                            } else { // axis == 2
                                io = i + nghost + offset1;
                                jo = j + nghost + offset2;
                                ko = k + nghost - 1;
                            }
                            left_cons.rho = interpolate_face_center(_u_.data(0, io, jo, ko), false, p, axis, nx, ny, nz);
                            left_cons.mx  = interpolate_face_center(_u_.data(1, io, jo, ko), false, p, axis, nx, ny, nz);
                            left_cons.my  = interpolate_face_center(_u_.data(2, io, jo, ko), false, p, axis, nx, ny, nz);
                            left_cons.mz  = interpolate_face_center(_u_.data(3, io, jo, ko), false, p, axis, nx, ny, nz);
                            left_cons.E   = interpolate_face_center(_u_.data(4, io, jo, ko), false, p, axis, nx, ny, nz);
                            cons_to_prim(left_cons, left_prim, gamma, isothermal, iso_cs);

                            // Interpolate node to the right of the interface
                            if (axis == 0) {
                                io = i + nghost;
                                jo = j + nghost + offset1;
                                ko = k + nghost + offset2;
                            } else if (axis == 1) {
                                io = i + nghost + offset1;
                                jo = j + nghost;
                                ko = k + nghost + offset2;
                            } else { // axis == 2
                                io = i + nghost + offset1;
                                jo = j + nghost + offset2;
                                ko = k + nghost;
                            }
                            right_cons.rho = interpolate_face_center(_u_.data(0, io, jo, ko), true, p, axis, nx, ny, nz);
                            right_cons.mx  = interpolate_face_center(_u_.data(1, io, jo, ko), true, p, axis, nx, ny, nz);
                            right_cons.my  = interpolate_face_center(_u_.data(2, io, jo, ko), true, p, axis, nx, ny, nz);
                            right_cons.mz  = interpolate_face_center(_u_.data(3, io, jo, ko), true, p, axis, nx, ny, nz);
                            right_cons.E   = interpolate_face_center(_u_.data(4, io, jo, ko), true, p, axis, nx, ny, nz);
                            cons_to_prim(right_cons, right_prim, gamma, isothermal, iso_cs);

                            // Call Riemann solver and store nodal fluxes
                            hllc_flux(left_prim, right_prim, flux, axis + 1, gamma, isothermal, iso_cs);
                            nodal_fluxes[0 * nnodes_max * nnodes_max + q1 * nnodes_max + q2] = flux.rho;
                            nodal_fluxes[1 * nnodes_max * nnodes_max + q1 * nnodes_max + q2] = flux.mx;
                            nodal_fluxes[2 * nnodes_max * nnodes_max + q1 * nnodes_max + q2] = flux.my;
                            nodal_fluxes[3 * nnodes_max * nnodes_max + q1 * nnodes_max + q2] = flux.mz;
                            nodal_fluxes[4 * nnodes_max * nnodes_max + q1 * nnodes_max + q2] = flux.E;
                        }
                    }

                    // Integrate nodal fluxes to get face-centered flux
                    ptrdiff_t center_offset = nnodes / 2;
                    if (axis == 0) {
                        for (int v = 0; v < nvars; ++v) {
                            F.mutable_at(v, i, j, k) = apply_2d_stencil(&nodal_fluxes[v * nnodes_max * nnodes_max + center_offset * nnodes_max + center_offset], stencil, stencil, temp, 0, 1, nnodes_max, 1, nnodes, nnodes);
                        }
                    } else if (axis == 1) {
                        for (int v = 0; v < nvars; ++v) {
                            G.mutable_at(v, i, j, k) = apply_2d_stencil(&nodal_fluxes[v * nnodes_max * nnodes_max + center_offset * nnodes_max + center_offset], stencil, stencil, temp, 0, 1, nnodes_max, 1, nnodes, nnodes);
                        }
                    } else {
                        for (int v = 0; v < nvars; ++v) {
                            H.mutable_at(v, i, j, k) = apply_2d_stencil(&nodal_fluxes[v * nnodes_max * nnodes_max + center_offset * nnodes_max + center_offset], stencil, stencil, temp, 0, 1, nnodes_max, 1, nnodes, nnodes);
                        }
                    }
                }
            }
        }
    }
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
    const int nx = static_cast<int>(_u_.shape(1));
    const int ny = static_cast<int>(_u_.shape(2));
    const int nz = static_cast<int>(_u_.shape(3));
    const int ndim = count_ndim(nx, ny, nz);

    if (ndim == 1) {
        update_1D_fv_fluxes(_u_, F, p, nghost, gamma, isothermal, iso_cs);
    } else if (ndim == 2) {
        update_2D_fv_fluxes(_u_, F, G, p, nghost, gamma, isothermal, iso_cs);
    } else if (ndim == 3) {
        update_3D_fv_fluxes(_u_, F, G, H, p, nghost, gamma, isothermal, iso_cs);
    } else {
        throw std::runtime_error("Invalid number of dimensions");
    }
}

PYBIND11_MODULE(_finite_volume_driver, m) {
    m.def("update_fv_fluxes", &update_fv_fluxes);
}
