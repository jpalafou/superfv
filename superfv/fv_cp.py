from typing import Literal, Optional, Tuple

from .axes import DIM_TO_AXIS
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.slicing import crop, replace_slice

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    p1_l_interpolation_kernel = cp.ElementwiseKernel(
        in_params="float64 ul, float64 uc, float64 ur",
        out_params="float64 uface",
        operation="uface = (ul + 4.0 * uc + -1.0 * ur) / 4.0;",
        name="p1_l_interpolation_kernel",
        no_return=True,
    )

    p1_r_interpolation_kernel = cp.ElementwiseKernel(
        in_params="float64 ul, float64 uc, float64 ur",
        out_params="float64 uface",
        operation="uface = (-1.0 * ul + 4.0 * uc + ur) / 4.0;",
        name="p1_r_interpolation_kernel",
        no_return=True,
    )

    p2_l_interpolation_kernel = cp.ElementwiseKernel(
        in_params="float64 ul, float64 uc, float64 ur",
        out_params="float64 uface",
        operation="uface = (2.0 * ul + 5.0 * uc + -1.0 * ur) / 6.0;",
        name="p2_l_interpolation_kernel",
        no_return=True,
    )

    p2_r_interpolation_kernel = cp.ElementwiseKernel(
        in_params="float64 ul, float64 uc, float64 ur",
        out_params="float64 uface",
        operation="uface = (-1.0 * ul + 5.0 * uc + 2.0 * ur) / 6.0;",
        name="p2_r_interpolation_kernel",
        no_return=True,
    )

    p3_l_interpolation_kernel = cp.ElementwiseKernel(
        in_params="float64 ul2, float64 ul1, float64 uc, float64 ur1, float64 ur2",
        out_params="float64 uface",
        operation="""
            uface = (-1.0 * ul2 + 10.0 * ul1 + 20.0 * uc + -6.0 * ur1 + 1.0 * ur2)
                / 24.0;
        """,
        name="p3_l_interpolation_kernel",
        no_return=True,
    )

    p3_r_interpolation_kernel = cp.ElementwiseKernel(
        in_params=("float64 ul2, float64 ul1, float64 uc, float64 ur1, float64 ur2"),
        out_params="float64 uface",
        operation="""
            uface = (1.0 * ul2 + -6.0 * ul1 + 20.0 * uc + 10.0 * ur1 + -1.0 * ur2)
                / 24.0;
        """,
        name="p3_r_interpolation_kernel",
        no_return=True,
    )

    p3_c_interpolation_kernel = cp.ElementwiseKernel(
        in_params="float64 ul, float64 uc, float64 ur",
        out_params="float64 ucenter",
        operation="""
            ucenter = ( -1.0 * ul + 26.0 * uc + -1.0 * ur ) / 24.0;
        """,
        name="p3_c_interpolation_kernel",
        no_return=True,
    )

    p3_integration_kernel = cp.ElementwiseKernel(
        in_params="float64 ul, float64 uc, float64 ur",
        out_params="float64 uaverage",
        operation="""
            uaverage = (ul + 22.0 * uc + ur ) / 24.0;
        """,
        name="p3_integration_kernel",
        no_return=True,
    )

    p4_l_interpolation_kernel = cp.ElementwiseKernel(
        in_params="float64 ul2, float64 ul1, float64 uc, float64 ur1, float64 ur2",
        out_params="float64 uface",
        operation="""
            uface = (-3.0 * ul2 + 27.0 * ul1 + 47.0 * uc + -13.0 * ur1 + 2.0 * ur2)
                / 60.0;
        """,
        name="p4_l_interpolation_kernel",
        no_return=True,
    )

    p4_r_interpolation_kernel = cp.ElementwiseKernel(
        in_params="float64 ul2, float64 ul1, float64 uc, float64 ur1, float64 ur2",
        out_params="float64 uface",
        operation="""
            uface = (2.0 * ul2 + -13.0 * ul1 + 47.0 * uc + 27.0 * ur1 + -3.0 * ur2)
                / 60.0;
        """,
        name="p4_r_interpolation_kernel",
        no_return=True,
    )

    p5_l_interpolation_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 ul3, float64 ul2, float64 ul1, float64 uc, float64 ur1, "
            "float64 ur2, float64 ur3"
        ),
        out_params="float64 uface",
        operation="""
            uface = (
                ul3
                + -10.0 * ul2
                + 59.0 * ul1
                + 94.0 * uc
                + -31.0 * ur1
                + 8.0 * ur2
                + -1.0 * ur3
            ) / 120.0;
        """,
        name="p5_l_interpolation_kernel",
        no_return=True,
    )

    p5_r_interpolation_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 ul3, float64 ul2, float64 ul1, float64 uc, float64 ur1, "
            "float64 ur2, float64 ur3"
        ),
        out_params="float64 uface",
        operation="""
            uface = (
                -1.0 * ul3
                + 8.0 * ul2
                + -31.0 * ul1
                + 94.0 * uc
                + 59.0 * ur1
                + -10.0 * ur2
                + ur3
            ) / 120.0;
        """,
        name="p5_r_interpolation_kernel",
        no_return=True,
    )

    p5_c_interpolation_kernel = cp.ElementwiseKernel(
        in_params="float64 ul2, float64 ul1, float64 uc, float64 ur1, float64 ur2",
        out_params="float64 ucenter",
        operation="""
            ucenter = (
                9.0 * ul2 + -116.0 * ul1 + 2134.0 * uc + -116.0 * ur1 + 9.0 * ur2
            ) / 1920.0;
        """,
        name="p5_c_interpolation_kernel",
        no_return=True,
    )

    p5_integration_kernel = cp.ElementwiseKernel(
        in_params="float64 ul2, float64 ul1, float64 uc, float64 ur1, float64 ur2",
        out_params="float64 uaverage",
        operation="""
            uaverage = (
                -17.0 * ul2 + 308.0 * ul1 + 5178.0 * uc + 308.0 * ur1 + -17.0 * ur2
            ) / 5760.0;
        """,
        name="p5_integration_kernel",
        no_return=True,
    )

    p6_l_interpolation_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 ul3, float64 ul2, float64 ul1, float64 uc, float64 ur1, "
            "float64 ur2, float64 ur3"
        ),
        out_params="float64 uface",
        operation="""
            uface = (
                4.0 * ul3
                + -38.0 * ul2
                + 214.0 * ul1
                + 319.0 * uc
                + -101.0 * ur1
                + 25.0 * ur2
                + -3.0 * ur3
            ) / 420.0;
        """,
        name="p6_l_interpolation_kernel",
        no_return=True,
    )

    p6_r_interpolation_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 ul3, float64 ul2, float64 ul1, float64 uc, float64 ur1, "
            "float64 ur2, float64 ur3"
        ),
        out_params="float64 uface",
        operation="""
            uface = (
                -3.0 * ul3
                + 25.0 * ul2
                + -101.0 * ul1
                + 319.0 * uc
                + 214.0 * ur1
                + -38.0 * ur2
                + 4.0 * ur3
            ) / 420.0;
        """,
        name="p6_r_interpolation_kernel",
        no_return=True,
    )

    p7_l_interpolation_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 ul4, float64 ul3, float64 ul2, float64 ul1, float64 uc, "
            "float64 ur1, float64 ur2, float64 ur3, float64 ur4"
        ),
        out_params="float64 uface",
        operation="""
            uface = (
                -3.0 * ul4
                + 34.0 * ul3
                + -194.0 * ul2
                + 898.0 * ul1
                + 1276.0 * uc
                + -446.0 * ur1
                + 142.0 * ur2
                + -30.0 * ur3
                + 3.0 * ur4
            ) / 1680.0;
        """,
        name="p7_l_interpolation_kernel",
        no_return=True,
    )

    p7_r_interpolation_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 ul4, float64 ul3, float64 ul2, float64 ul1, float64 uc, "
            "float64 ur1, float64 ur2, float64 ur3, float64 ur4"
        ),
        out_params="float64 uface",
        operation="""
            uface = (
                3.0 * ul4
                + -30.0 * ul3
                + 142.0 * ul2
                + -446.0 * ul1
                + 1276.0 * uc
                + 898.0 * ur1
                + -194.0 * ur2
                + 34.0 * ur3
                + -3.0 * ur4
            ) / 1680.0;
        """,
        name="p7_r_interpolation_kernel",
        no_return=True,
    )

    p7_c_interpolation_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 ul3, float64 ul2, float64 ul1, float64 uc, float64 ur1, "
            "float64 ur2, float64 ur3"
        ),
        out_params="float64 ucenter",
        operation="""
            ucenter = (
                -75.0 * ul3
                + 954.0 * ul2
                + -7621.0 * ul1
                + 121004.0 * uc
                + -7621.0 * ur1
                + 954.0 * ur2
                + -75.0 * ur3
            ) / 107520.0;
        """,
        name="p7_c_interpolation_kernel",
        no_return=True,
    )

    p7_integration_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 ul3, float64 ul2, float64 ul1, float64 uc, float64 ur1, "
            "float64 ur2, float64 ur3"
        ),
        out_params="float64 uaverage",
        operation="""
            uaverage = (
                367.0 * ul3
                + -5058.0 * ul2
                + 57249.0 * ul1
                + 862564.0 * uc
                + 57249.0 * ur1
                + -5058.0 * ur2
                + 367.0 * ur3
            ) / 967680.0;
        """,
        name="p7_integration_kernel",
        no_return=True,
    )


def sweep_kernel_helper(
    u: ArrayLike, axis: int, p: int, pos: int, integrate: bool, *, out: ArrayLike
):
    """
    Helper function to dispatch to the appropriate interpolation/integration kernel.

    Args:
        u: CuPy array of shape (nvars, nx, ny, nz).
        axis: Axis along which to perform the operation (1 for x, 2 for y, 3 for z).
        p: Polynomial order (0 to 7).
        pos: Position indicator (-1 for left, 0 for center, 1 for right).
        integrate: If True, perform integration instead of interpolation and ignore
            pos.
        out: CuPy array to store the output, of shape (nvars, nx, ny, nz, >=1).
    """

    if pos not in (-1, 0, 1):
        raise ValueError("pos must be -1, 0, or 1")

    match p:
        case 0:
            out[..., 0] = u
        case 1:
            ul = u[crop(axis, (None, -2), ndim=4)]
            uc = u[crop(axis, (1, -1), ndim=4)]
            ur = u[crop(axis, (2, None), ndim=4)]
            inner = out[crop(axis, (1, -1))][..., 0]

            if integrate:
                out[..., 0] = u
            elif pos == -1:
                p1_l_interpolation_kernel(ul, uc, ur, inner)
            elif pos == 1:
                p1_r_interpolation_kernel(ul, uc, ur, inner)
            else:  # pos == 0
                out[..., 0] = u

        case 2:
            ul = u[crop(axis, (None, -2), ndim=4)]
            uc = u[crop(axis, (1, -1), ndim=4)]
            ur = u[crop(axis, (2, None), ndim=4)]
            inner = out[replace_slice(crop(axis, (1, -1), ndim=5), 4, 0)]

            if integrate:
                p3_integration_kernel(ul, uc, ur, inner)
            elif pos == -1:
                p2_l_interpolation_kernel(ul, uc, ur, inner)
            elif pos == 1:
                p2_r_interpolation_kernel(ul, uc, ur, inner)
            else:  # pos == 0
                p3_c_interpolation_kernel(ul, uc, ur, inner)

        case 3:
            ul2 = u[crop(axis, (None, -4), ndim=4)]
            ul1 = u[crop(axis, (1, -3), ndim=4)]
            uc = u[crop(axis, (2, -2), ndim=4)]
            ur1 = u[crop(axis, (3, -1), ndim=4)]
            ur2 = u[crop(axis, (4, None), ndim=4)]
            inner = out[replace_slice(crop(axis, (2, -2), ndim=5), 4, 0)]

            if integrate:
                p3_integration_kernel(ul1, uc, ur1, inner)
            elif pos == -1:
                p3_l_interpolation_kernel(ul2, ul1, uc, ur1, ur2, inner)
            elif pos == 1:
                p3_r_interpolation_kernel(ul2, ul1, uc, ur1, ur2, inner)
            else:  # pos == 0
                p3_c_interpolation_kernel(ul1, uc, ur1, inner)

        case 4:
            ul2 = u[crop(axis, (None, -4), ndim=4)]
            ul1 = u[crop(axis, (1, -3), ndim=4)]
            uc = u[crop(axis, (2, -2), ndim=4)]
            ur1 = u[crop(axis, (3, -1), ndim=4)]
            ur2 = u[crop(axis, (4, None), ndim=4)]
            inner = out[replace_slice(crop(axis, (2, -2), ndim=5), 4, 0)]

            if integrate:
                p5_integration_kernel(ul2, ul1, uc, ur1, ur2, inner)
            elif pos == -1:
                p4_l_interpolation_kernel(ul2, ul1, uc, ur1, ur2, inner)
            elif pos == 1:
                p4_r_interpolation_kernel(ul2, ul1, uc, ur1, ur2, inner)
            else:  # pos == 0
                p5_c_interpolation_kernel(ul2, ul1, uc, ur1, ur2, inner)

        case 5:
            ul3 = u[crop(axis, (None, -6), ndim=4)]
            ul2 = u[crop(axis, (1, -5), ndim=4)]
            ul1 = u[crop(axis, (2, -4), ndim=4)]
            uc = u[crop(axis, (3, -3), ndim=4)]
            ur1 = u[crop(axis, (4, -2), ndim=4)]
            ur2 = u[crop(axis, (5, -1), ndim=4)]
            ur3 = u[crop(axis, (6, None), ndim=4)]
            inner = out[replace_slice(crop(axis, (3, -3), ndim=5), 4, 0)]

            if integrate:
                p5_integration_kernel(ul2, ul1, uc, ur1, ur2, inner)
            elif pos == -1:
                p5_l_interpolation_kernel(ul3, ul2, ul1, uc, ur1, ur2, ur3, inner)
            elif pos == 1:
                p5_r_interpolation_kernel(ul3, ul2, ul1, uc, ur1, ur2, ur3, inner)
            else:  # pos == 0
                p5_c_interpolation_kernel(ul2, ul1, uc, ur1, ur2, inner)

        case 6:
            ul3 = u[crop(axis, (None, -6), ndim=4)]
            ul2 = u[crop(axis, (1, -5), ndim=4)]
            ul1 = u[crop(axis, (2, -4), ndim=4)]
            uc = u[crop(axis, (3, -3), ndim=4)]
            ur1 = u[crop(axis, (4, -2), ndim=4)]
            ur2 = u[crop(axis, (5, -1), ndim=4)]
            ur3 = u[crop(axis, (6, None), ndim=4)]
            inner = out[replace_slice(crop(axis, (3, -3), ndim=5), 4, 0)]

            if integrate:
                p7_integration_kernel(ul3, ul2, ul1, uc, ur1, ur2, ur3, inner)
            elif pos == -1:
                p6_l_interpolation_kernel(ul3, ul2, ul1, uc, ur1, ur2, ur3, inner)
            elif pos == 1:
                p6_r_interpolation_kernel(ul3, ul2, ul1, uc, ur1, ur2, ur3, inner)
            else:  # pos == 0
                p7_c_interpolation_kernel(ul3, ul2, ul1, uc, ur1, ur2, ur3, inner)

        case 7:
            ul4 = u[crop(axis, (None, -8), ndim=4)]
            ul3 = u[crop(axis, (1, -7), ndim=4)]
            ul2 = u[crop(axis, (2, -6), ndim=4)]
            ul1 = u[crop(axis, (3, -5), ndim=4)]
            uc = u[crop(axis, (4, -4), ndim=4)]
            ur1 = u[crop(axis, (5, -3), ndim=4)]
            ur2 = u[crop(axis, (6, -2), ndim=4)]
            ur3 = u[crop(axis, (7, -1), ndim=4)]
            ur4 = u[crop(axis, (8, None), ndim=4)]
            inner = out[replace_slice(crop(axis, (4, -4), ndim=5), 4, 0)]

            if integrate:
                p7_integration_kernel(ul3, ul2, ul1, uc, ur1, ur2, ur3, inner)
            elif pos == -1:
                p7_l_interpolation_kernel(
                    ul4, ul3, ul2, ul1, uc, ur1, ur2, ur3, ur4, inner
                )
            elif pos == 1:
                p7_r_interpolation_kernel(
                    ul4, ul3, ul2, ul1, uc, ur1, ur2, ur3, ur4, inner
                )
            else:  # pos == 0
                p7_c_interpolation_kernel(ul3, ul2, ul1, uc, ur1, ur2, ur3, inner)

        case _:
            raise ValueError(
                f"Unsupported polynomial order for interpolation/integration {p=}"
            )


def interpolation_kernel_helper_1d(
    u: ArrayLike,
    p: int,
    dim: Literal["x", "y", "z"],
    center: bool,
    integrate: bool,
    *,
    out: ArrayLike,
):
    """
    Helper function to perform 1D interpolation/integration along a specified
    dimension.

    Args:
        u: CuPy array of shape (nvars, nx, ny, nz).
        p: Polynomial order (0 to 7).
        dim: Dimension along which to interpolate ('x', 'y', or 'z').
        center: If True, perform center interpolation and write to out[..., 0]. If
            False, perform left/right interpolation and write to out[..., 0] and
            out[..., 1], respectively.
        integrate: If True, perform integration instead of interpolation and ignore
            center.
        out: CuPy array to store the output, of shape (nvars, nx, ny, nz, >=1).
    """

    axis = DIM_TO_AXIS[dim]

    if integrate:
        sweep_kernel_helper(u, axis, p, pos=0, integrate=True, out=out)
    elif center:
        sweep_kernel_helper(u, axis, p, pos=0, integrate=False, out=out)
    else:
        sweep_kernel_helper(u, axis, p, pos=-1, integrate=False, out=out[..., :1])
        sweep_kernel_helper(u, axis, p, pos=1, integrate=False, out=out[..., 1:2])


def interpolation_kernel_helper_2d(
    u: ArrayLike,
    p: int,
    dim1: Literal["x", "y", "z"],
    dim2: Literal["x", "y", "z"],
    center: bool,
    integrate: bool,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
):
    """
    Helper function to perform 2D interpolation/integration along two specified
    dimensions.

    Args:
        u: CuPy array of shape (nvars, nx, ny, nz).
        p: Polynomial order (0 to 7).
        dim1: First dimension along which to interpolate ('x', 'y', or 'z').
        dim2: Second dimension along which to interpolate ('x', 'y', or 'z').
        center: If True, perform center interpolation and write to out[..., 0]. If
            False, perform left/right interpolation and write to out[..., 0] and
            out[..., 1], respectively.
        integrate: If True, perform integration instead of interpolation and ignore
            center.
        out: CuPy array to store the output, of shape (nvars, nx, ny, nz, >=1).
        buffer: CuPy array for intermediate storage, of shape (nvars, nx, ny, nz, >=1).
    """

    axis1 = DIM_TO_AXIS[dim1]
    axis2 = DIM_TO_AXIS[dim2]

    if integrate and center:
        sweep_kernel_helper(u, axis2, p, pos=0, integrate=True, out=buffer)
        sweep_kernel_helper(buffer[..., 0], axis1, p, pos=0, integrate=True, out=out)
    elif integrate:
        sweep_kernel_helper(u, axis2, p, pos=0, integrate=True, out=out)
    elif center:
        sweep_kernel_helper(u, axis2, p, pos=0, integrate=False, out=buffer)
        sweep_kernel_helper(buffer[..., 0], axis1, p, pos=0, integrate=False, out=out)
    else:
        sweep_kernel_helper(u, axis2, p, pos=0, integrate=False, out=buffer)
        sweep_kernel_helper(
            buffer[..., 0], axis1, p, pos=-1, integrate=False, out=out[..., :1]
        )
        sweep_kernel_helper(
            buffer[..., 0], axis1, p, pos=1, integrate=False, out=out[..., 1:2]
        )


def interpolation_kernel_helper(
    u: ArrayLike,
    face_dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    center: bool,
    integrate: bool,
    *,
    out: ArrayLike,
    buffer: Optional[ArrayLike] = None,
):
    """
    Helper function to dispatch to the appropriate interpolation/integration kernel
    based on the number of active dimensions.

    Args:
        u: CuPy array of shape (nvars, nx, ny, nz).
        face_dim: Dimension along which to interpolate ('x', 'y', or 'z').
        active_dims: Tuple of active dimensions ('x', 'y', 'z') for interpolation.
            Must contain face_dim.
        p: Polynomial order (0 to 7).
        center: If True, perform center interpolation and write to out[..., 0]. If
            False, perform left/right interpolation and write to out[..., 0] and
            out[..., 1], respectively.
        integrate: If True, perform integration instead of interpolation and ignore
            center.
        out: CuPy array to store the output, of shape (nvars, nx, ny, nz, >=1).
        buffer: Optional CuPy array for intermediate storage when active_dims has
            length 2 or more. Must be provided in that case.
    """
    if len(active_dims) == 1:
        interpolation_kernel_helper_1d(
            u,
            p,
            face_dim,
            center=center,
            integrate=integrate,
            out=out,
        )
        return

    if buffer is None:
        raise ValueError(
            "buffer must be provided for 2 or more interpolation dimensions."
        )

    if len(active_dims) == 2:
        if center:
            dim1 = active_dims[0]
            dim2 = active_dims[1]
        else:
            dim1 = face_dim
            dim2 = next(dim for dim in active_dims if dim != face_dim)

        interpolation_kernel_helper_2d(
            u,
            p,
            dim1,
            dim2,
            center=center,
            integrate=integrate,
            out=out,
            buffer=buffer,
        )
        return
    elif len(active_dims) == 3:
        # 3D interpolation/integration not yet implemented
        raise NotImplementedError("3D interpolation/integration not yet implemented.")
    else:
        raise ValueError("active_dims must have length 1, 2, or 3.")
