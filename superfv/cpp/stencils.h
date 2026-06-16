#pragma once

#include <stdexcept>
#include "constants.h"

struct Stencil {
    double weights[MAX_NODES];
    int n;
};

void validate_stencil(const Stencil& stencil) {
    if (stencil.n < 1 || stencil.n > MAX_NODES) {
        throw std::invalid_argument("Stencil size is out of bounds");
    }
}

Stencil conservative_interpolation_of_cell_center(const int p) {
    Stencil out{};

    if (p == 0 or p == 1) {
        out.weights[0] = 1.0;
        out.n = 1;
    } else if (p == 2 or p == 3) {
        out.weights[0] = -1.0 / 24.0;
        out.weights[1] = 13.0 / 12.0;
        out.weights[2] = -1.0 / 24.0;
        out.n = 3;
    } else if (p == 4 or p == 5) {
        out.weights[0] = 3.0 / 640.0;
        out.weights[1] = -29.0 / 480.0;
        out.weights[2] = 1067.0 / 960.0;
        out.weights[3] = -29.0 / 480.0;
        out.weights[4] = 3.0 / 640.0;
        out.n = 5;
    } else if (p == 6 or p == 7) {
        out.weights[0] = -5.0 / 7168.0;
        out.weights[1] = 159.0 / 17920.0;
        out.weights[2] = -7621.0 / 107520.0;
        out.weights[3] = 30251.0 / 26880.0;
        out.weights[4] = -7621.0 / 107520.0;
        out.weights[5] = 159.0 / 17920.0;
        out.weights[6] = -5.0 / 7168.0;
        out.n = 7;
    } else {
        throw std::invalid_argument("Invalid order of interpolation");
    }
    validate_stencil(out);
    return out;
}

Stencil transverse_integration_of_cell_average(const int p) {
    Stencil out{};

    if (p == 0 or p == 1) {
        out.weights[0] = 1.0;
        out.n = 1;
    } else if (p == 2 or p == 3) {
        out.weights[0] = 1.0 / 24.0;
        out.weights[1] = 11.0 / 12.0;
        out.weights[2] = 1.0 / 24.0;
        out.n = 3;
    } else if (p == 4 or p == 5) {
        out.weights[0] = -17.0 / 5760.0;
        out.weights[1] = 77.0 / 1440.0;
        out.weights[2] = 863.0 / 960.0;
        out.weights[3] = 77.0 / 1440.0;
        out.weights[4] = -17.0 / 5760.0;
        out.n = 5;
    } else if (p == 6 or p == 7) {
        out.weights[0] = 367.0 / 967680.0;
        out.weights[1] = -281.0 / 53760.0;
        out.weights[2] = 6361.0 / 107520.0;
        out.weights[3] = 215641.0 / 241920.0;
        out.weights[4] = 6361.0 / 107520.0;
        out.weights[5] = -281.0 / 53760.0;
        out.weights[6] = 367.0 / 967680.0;
        out.n = 7;
    } else {
        throw std::invalid_argument("Invalid order of interpolation");
    }
    validate_stencil(out);
    return out;
}

Stencil conservative_interpolation_of_left_or_right_face(const int p, const bool left) {
    Stencil out{};

    switch (p) {
        case 0:
            out.weights[0] = 1.0;
            out.n = 1;
            break;
        case 1:
            if (left) {
                out.weights[0] = 1.0 / 4.0;
                out.weights[1] = 1.0;
                out.weights[2] = -1.0 / 4.0;
            } else {
                out.weights[0] = -1.0 / 4.0;
                out.weights[1] = 1.0;
                out.weights[2] = 1.0 / 4.0;
            }
            out.n = 3;
            break;
        case 2:
            if (left) {
                out.weights[0] = 1.0 / 3.0;
                out.weights[1] = 5.0 / 6.0;
                out.weights[2] = -1.0 / 6.0;
            } else {
                out.weights[0] = -1.0 / 6.0;
                out.weights[1] = 5.0 / 6.0;
                out.weights[2] = 1.0 / 3.0;
            }
            out.n = 3;
            break;
        case 3:
            if (left) {
                out.weights[0] = -1.0 / 24.0;
                out.weights[1] = 5.0 / 12.0;
                out.weights[2] = 5.0 / 6.0;
                out.weights[3] = -1.0 / 4.0;
                out.weights[4] = 1.0 / 24.0;
            } else {
                out.weights[0] = 1.0 / 24.0;
                out.weights[1] = -1.0 / 4.0;
                out.weights[2] = 5.0 / 6.0;
                out.weights[3] = 5.0 / 12.0;
                out.weights[4] = -1.0 / 24.0;
            }
            out.n = 5;
            break;
        case 4:
            if (left) {
                out.weights[0] = -1.0 / 20.0;
                out.weights[1] = 9.0 / 20.0;
                out.weights[2] = 47.0 / 60.0;
                out.weights[3] = -13.0 / 60.0;
                out.weights[4] = 1.0 / 30.0;
            } else {
                out.weights[0] = 1.0 / 30.0;
                out.weights[1] = -13.0 / 60.0;
                out.weights[2] = 47.0 / 60.0;
                out.weights[3] = 9.0 / 20.0;
                out.weights[4] = -1.0 / 20.0;
            }
            out.n = 5;
            break;
        case 5:
            if (left) {
                out.weights[0] = 1.0 / 120.0;
                out.weights[1] = -1.0 / 12.0;
                out.weights[2] = 59.0 / 120.0;
                out.weights[3] = 47.0 / 60.0;
                out.weights[4] = -31.0 / 120.0;
                out.weights[5] = 1.0 / 15.0;
                out.weights[6] = -1.0 / 120.0;
            } else {
                out.weights[0] = -1.0 / 120.0;
                out.weights[1] = 1.0 / 15.0;
                out.weights[2] = -31.0 / 120.0;
                out.weights[3] = 47.0 / 60.0;
                out.weights[4] = 59.0 / 120.0;
                out.weights[5] = -1.0 / 12.0;
                out.weights[6] = 1.0 / 120.0;
            }
            out.n = 7;
            break;
        case 6:
            if (left) {
                out.weights[0] = 1.0 / 105.0;
                out.weights[1] = -19.0 / 210.0;
                out.weights[2] = 107.0 / 210.0;
                out.weights[3] = 319.0 / 420.0;
                out.weights[4] = -101.0 / 420.0;
                out.weights[5] = 5.0 / 84.0;
                out.weights[6] = -1.0 / 140.0;
            } else {
                out.weights[0] = -1.0 / 140.0;
                out.weights[1] = 5.0 / 84.0;
                out.weights[2] = -101.0 / 420.0;
                out.weights[3] = 319.0 / 420.0;
                out.weights[4] = 107.0 / 210.0;
                out.weights[5] = -19.0 / 210.0;
                out.weights[6] = 1.0 / 105.0;
            }
            out.n = 7;
            break;
        case 7:
            if (left) {
                out.weights[0] = -1.0 / 560.0;
                out.weights[1] = 17.0 / 840.0;
                out.weights[2] = -97.0 / 840.0;
                out.weights[3] = 449.0 / 840.0;
                out.weights[4] = 319.0 / 420.0;
                out.weights[5] = -223.0 / 840.0;
                out.weights[6] = 71.0 / 840.0;
                out.weights[7] = -1.0 / 56.0;
                out.weights[8] = 1.0 / 560.0;
            } else {
                out.weights[0] = 1.0 / 560.0;
                out.weights[1] = -1.0 / 56.0;
                out.weights[2] = 71.0 / 840.0;
                out.weights[3] = -223.0 / 840.0;
                out.weights[4] = 319.0 / 420.0;
                out.weights[5] = 449.0 / 840.0;
                out.weights[6] = -97.0 / 840.0;
                out.weights[7] = 17.0 / 840.0;
                out.weights[8] = -1.0 / 560.0;
            }
            out.n = 9;
            break;
        default:
            throw std::invalid_argument("Invalid order of interpolation");
    }
    validate_stencil(out);
    return out;
}
