#include <cstddef>
#include <stdexcept>

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
    const int nweights
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
