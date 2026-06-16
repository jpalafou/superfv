#pragma once

#include <stdexcept>

int write_weights_for_conservative_interpolation_of_cell_center(
    double* stencil,
    const size_t stencil_size,
    const int p
) {
    if (stencil == nullptr) {
        throw std::invalid_argument("Stencil pointer cannot be null");
    }
    if (stencil_size < 7) {
        throw std::invalid_argument("Stencil size must be at least 7");
    }

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
    return nkernel;
}

int write_weights_for_transverse_integration_of_cell_average(
    double* stencil,
    const size_t stencil_size,
    const int p
) {
    if (stencil == nullptr) {
        throw std::invalid_argument("Stencil pointer cannot be null");
    }
    if (stencil_size < 7) {
        throw std::invalid_argument("Stencil size must be at least 7");
    }

    int nkernel = 0;
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
    return nkernel;
}

int write_weights_for_conservative_interpolation_of_left_or_right_face(
    double* stencil,
    const size_t stencil_size,
    const int p,
    const bool left
) {
    if (stencil == nullptr) {
        throw std::invalid_argument("Stencil pointer cannot be null");
    }
    if (stencil_size < 9) {
        throw std::invalid_argument("Stencil size must be at least 9");
    }

    int nkernel = 0;
    switch (p) {
        case 0:
            stencil[0] = 1.0;
            nkernel = 1;
            break;
        case 1:
            if (left) {
                stencil[0] = 1.0 / 4.0;
                stencil[1] = 1.0;
                stencil[2] = -1.0 / 4.0;
            } else {
                stencil[0] = -1.0 / 4.0;
                stencil[1] = 1.0;
                stencil[2] = 1.0 / 4.0;
            }
            nkernel = 3;
            break;
        case 2:
            if (left) {
                stencil[0] = 1.0 / 3.0;
                stencil[1] = 5.0 / 6.0;
                stencil[2] = -1.0 / 6.0;
            } else {
                stencil[0] = -1.0 / 6.0;
                stencil[1] = 5.0 / 6.0;
                stencil[2] = 1.0 / 3.0;
            }
            nkernel = 3;
            break;
        case 3:
            if (left) {
                stencil[0] = -1.0 / 24.0;
                stencil[1] = 5.0 / 12.0;
                stencil[2] = 5.0 / 6.0;
                stencil[3] = -1.0 / 4.0;
                stencil[4] = 1.0 / 24.0;
            } else {
                stencil[0] = 1.0 / 24.0;
                stencil[1] = -1.0 / 4.0;
                stencil[2] = 5.0 / 6.0;
                stencil[3] = 5.0 / 12.0;
                stencil[4] = -1.0 / 24.0;
            }
            nkernel = 5;
            break;
        case 4:
            if (left) {
                stencil[0] = -1.0 / 20.0;
                stencil[1] = 9.0 / 20.0;
                stencil[2] = 47.0 / 60.0;
                stencil[3] = -13.0 / 60.0;
                stencil[4] = 1.0 / 30.0;
            } else {
                stencil[0] = 1.0 / 30.0;
                stencil[1] = -13.0 / 60.0;
                stencil[2] = 47.0 / 60.0;
                stencil[3] = 9.0 / 20.0;
                stencil[4] = -1.0 / 20.0;
            }
            nkernel = 5;
            break;
        case 5:
            if (left) {
                stencil[0] = 1.0 / 120.0;
                stencil[1] = -1.0 / 12.0;
                stencil[2] = 59.0 / 120.0;
                stencil[3] = 47.0 / 60.0;
                stencil[4] = -31.0 / 120.0;
                stencil[5] = 1.0 / 15.0;
                stencil[6] = -1.0 / 120.0;
            } else {
                stencil[0] = -1.0 / 120.0;
                stencil[1] = 1.0 / 15.0;
                stencil[2] = -31.0 / 120.0;
                stencil[3] = 47.0 / 60.0;
                stencil[4] = 59.0 / 120.0;
                stencil[5] = -1.0 / 12.0;
                stencil[6] = 1.0 / 120.0;
            }
            nkernel = 7;
            break;
        case 6:
            if (left) {
                stencil[0] = 1.0 / 105.0;
                stencil[1] = -19.0 / 210.0;
                stencil[2] = 107.0 / 210.0;
                stencil[3] = 319.0 / 420.0;
                stencil[4] = -101.0 / 420.0;
                stencil[5] = 5.0 / 84.0;
                stencil[6] = -1.0 / 140.0;
            } else {
                stencil[0] = -1.0 / 140.0;
                stencil[1] = 5.0 / 84.0;
                stencil[2] = -101.0 / 420.0;
                stencil[3] = 319.0 / 420.0;
                stencil[4] = 107.0 / 210.0;
                stencil[5] = -19.0 / 210.0;
                stencil[6] = 1.0 / 105.0;
            }
            nkernel = 7;
            break;
        case 7:
            if (left) {
                stencil[0] = -1.0 / 560.0;
                stencil[1] = 17.0 / 840.0;
                stencil[2] = -97.0 / 840.0;
                stencil[3] = 449.0 / 840.0;
                stencil[4] = 319.0 / 420.0;
                stencil[5] = -223.0 / 840.0;
                stencil[6] = 71.0 / 840.0;
                stencil[7] = -1.0 / 56.0;
                stencil[8] = 1.0 / 560.0;
            } else {
                stencil[0] = 1.0 / 560.0;
                stencil[1] = -1.0 / 56.0;
                stencil[2] = 71.0 / 840.0;
                stencil[3] = -223.0 / 840.0;
                stencil[4] = 319.0 / 420.0;
                stencil[5] = 449.0 / 840.0;
                stencil[6] = -97.0 / 840.0;
                stencil[7] = 17.0 / 840.0;
                stencil[8] = -1.0 / 560.0;
            }
            nkernel = 9;
            break;
        default:
            throw std::invalid_argument("Invalid order of interpolation");
    }
    return nkernel;
}
