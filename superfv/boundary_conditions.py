from typing import Callable, Literal, Optional, Tuple, Union, cast

from .fv import AXIS_TO_DIM
from .mesh import UniformFVMesh
from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap, crop

# define custom type annotations
Field = Callable[
    [VariableIndexMap, ArrayLike, ArrayLike, ArrayLike, Optional[float]],
    ArrayLike,
]
DirichletBC = Union[Field, Tuple[Optional[Field], Optional[Field]]]
FieldWrapper = Callable[[Field], Field]
BCs = Literal[
    "none", "periodic", "dirichlet", "free", "symmetric", "reflective", "zeros", "ones"
]
FieldFunction = Callable[
    [VariableIndexMap, ArrayLike, ArrayLike, ArrayLike, Optional[float]],
    ArrayLike,
]


def apply_bc(
    _u_: ArrayLike,
    pad_width: Tuple[int, int, int],
    mode: Tuple[Tuple[BCs, BCs], Tuple[BCs, BCs], Tuple[BCs, BCs]],
    dirichlet_mode: Optional[
        Literal["fv-averages", "cell-centers", "face-nodes"]
    ] = None,
    f: Tuple[
        Tuple[Optional[FieldFunction], Optional[FieldFunction]],
        Tuple[Optional[FieldFunction], Optional[FieldFunction]],
        Tuple[Optional[FieldFunction], Optional[FieldFunction]],
    ] = ((None, None), (None, None), (None, None)),
    variable_index_map: Optional[VariableIndexMap] = None,
    mesh: Optional[UniformFVMesh] = None,
    t: Optional[float] = None,
    face_dim: Optional[Literal["x", "y", "z"]] = None,
    face_pos: Optional[Literal["l", "r"]] = None,
    p: Optional[int] = None,
):
    """
    Apply boundary conditions to the array _u_.

    Args:
        _u_: Array to which the boundary conditions are applied. Expected to be 4D
            (nvars, nx, ny, nz) or 5D (nvars, nx, ny, nz, n_quadrature_points).
        pad_width: Tuple of integers indicating the padding width for each dimension:
            x (axis 1), y (axis 2), and z (axis 3).
        mode: Tuple of tuples specifying the boundary conditions for each dimension.
            Each inner tuple contains two strings for the left and right boundaries.
            Supported boundary condition names include:
            - "none": No boundary condition applied.
            - "periodic": Apply periodic boundary conditions.
            - "dirichlet": Apply Dirichlet boundary conditions.
        dirichlet_mode: Optional argument for when mode is "dirichlet". Specifies how
            the Dirichlet conditions are applied. Can be one of the following:
            - "fv-averages": Apply finite-volume averages of the Dirichlet function.
                _u_ is expected to be 4D (nvars, nx, ny, nz). p must be provided to
                perform the quadrature.
            - "cell-centers": Apply Dirichlet condition at cell centers. _u_ is
                expected to be 5D (nvars, nx, ny, nz, 1).
            - "face-nodes": Apply Dirichlet condition at face nodes. _u_ is expected to
                be 5D (nvars, nx, ny, nz, n_quadrature_points).
        f: Tuple of tuples containing functions defining the Dirichlet conditions for
            each dimension. Each inner tuple contains two functions for the left and
            right boundaries. If mode is "dirichlet", these functions must be provided,
            otherwise they can be None. Each function must accept the following
            arguments:
            - variable_index_map: VariableIndexMap object.
            - X: x-coordinate array. Has shape (nx, ny, nz).
            - Y: y-coordinate array. Has shape (nx, ny, nz).
            - Z: z-coordinate array. Has shape (nx, ny, nz).
            - t: Optional time at which the Dirichlet condition is applied.
        variable_index_map: Optional argument for when mode is "dirichlet" or
            "reflective".
        mesh: Optional argument for when mode is "dirichlet". If provided, it will be
            used to access mesh properties.
        t: Optional time at which the Dirichlet condition is applied. May be None if
            the Dirichlet function does not depend on time.
        face_dim: Optional argument for when dirichlet_mode is "face-nodes". If
            provided, specifies the face dimension for quadrature points. Can be one of
            "x", "y", or "z".
        face_pos: Optional argument for when dirichlet_mode is "face-nodes". If
            provided, specifies the face position for quadrature points. Can be one of
            "l" (left) or "r" (right).
        p: Optional argument for when mode is "dirichlet" and dirichlet_mode is
            "fv-averages" or "face-nodes". Specifies the polynomial degree of the
            quadrature rule used to evaluate the Dirichlet function.
    """
    for i, dim in enumerate(("x", "y", "z")):
        ip = i + 1
        pad_i = pad_width[i]
        for j, pos in enumerate(("l", "r")):
            left = not bool(j)
            if pad_i == 0:
                continue
            match mode[i][j]:
                case "none":
                    continue
                case "periodic":
                    apply_periodic_bc(_u_, pad_i, ip, left)
                case "dirichlet":
                    fij = f[i][j]
                    if dirichlet_mode is None:
                        raise ValueError(
                            "dirichlet_mode must be provided when applying Dirichlet BCs."
                        )
                    if fij is None:
                        raise ValueError(
                            "Function for Dirichlet condition must be provided."
                        )
                    if variable_index_map is None:
                        raise ValueError(
                            "VariableIndexMap must be provided for Dirichlet BCs."
                        )
                    if mesh is None:
                        raise ValueError(
                            "UniformFVMesh must be provided for Dirichlet BCs."
                        )
                    if dirichlet_mode in {"fv-averages", "face-nodes"} and p is None:
                        raise ValueError(
                            "Quadrature degree `p` must be provided for 'fv-averages' or 'face-nodes' mode."
                        )
                    apply_dirichlet_bc(
                        _u_,
                        pad_i,
                        ip,
                        left,
                        dirichlet_mode,
                        fij,
                        variable_index_map,
                        mesh,
                        t,
                        face_dim,
                        face_pos,
                        p,
                    )
                case "free":
                    apply_free_bc(_u_, pad_i, ip, left)
                case "symmetric":
                    apply_symmetric_bc(_u_, pad_i, ip, left)
                case "reflective":
                    if variable_index_map is None:
                        raise ValueError(
                            "VariableIndexMap must be provided for reflective BCs."
                        )
                    apply_reflective_bc(_u_, pad_i, ip, left, variable_index_map)
                case "zeros":
                    apply_uniform_bc(_u_, pad_i, ip, left, 0.0)
                case "ones":
                    apply_uniform_bc(_u_, pad_i, ip, left, 1.0)
                case _:
                    raise ValueError(
                        f"Boundary condition '{mode[i][j]}' not implemented for {dim}{pos} boundary."
                    )


def apply_periodic_bc(_u_: ArrayLike, slab_thickness: int, axis: int, left: bool):
    """
    Apply periodic boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        slab_thickness: Thickness of the slab (number of cells) to which the BC is
            applied.
        axis: Axis along which the BC is applied (1: x, 2: y, 3: z).
        left: Whether the BC is applied to the left (True) or right (False) boundary.
    """
    st = slab_thickness
    if left:
        outer_slice = crop(axis, (None, st))
        inner_slice = crop(axis, (-2 * st, -st))
    else:
        outer_slice = crop(axis, (-st, None))
        inner_slice = crop(axis, (st, 2 * st))
    _u_[outer_slice] = _u_[inner_slice]


def apply_dirichlet_bc(
    _u_: ArrayLike,
    slab_thickness: int,
    axis: int,
    left: bool,
    dirichlet_mode: Literal["fv-averages", "cell-centers", "face-nodes"],
    f: Field,
    variable_index_map: VariableIndexMap,
    mesh: UniformFVMesh,
    t: Optional[float],
    face_dim: Optional[Literal["x", "y", "z"]] = None,
    face_pos: Optional[Literal["l", "r"]] = None,
    p: Optional[int] = None,
):
    """
    Apply Dirichlet boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        slab_thickness: Thickness of the slab (number of cells) to which the BC is
            applied.
        axis: Axis along which the BC is applied (1: x, 2: y, 3: z).
        left: Whether the BC is applied to the left (True) or right (False) boundary.
        dirichlet_mode: Specifies how the Dirichlet conditions are applied. Can be one
            of the following:
            - "fv-averages": Apply finite-volume averages of the Dirichlet function.
                _u_ is expected to be 4D (nvars, nx, ny, nz). p must be provided to
                perform the quadrature.
            - "cell-centers": Apply Dirichlet condition at cell centers. _u_ is
                expected to be 5D (nvars, nx, ny, nz, 1).
            - "face-nodes": Apply Dirichlet condition at face nodes. _u_ is expected to
                be 5D (nvars, nx, ny, nz, n_quadrature_points).
        f: Functions defining the Dirichlet conditions for the specified slab. The
            function must accept the following arguments:
            - variable_index_map: VariableIndexMap object.
            - X: x-coordinate array. Has shape (nx, ny, nz).
            - Y: y-coordinate array. Has shape (nx, ny, nz).
            - Z: z-coordinate array. Has shape (nx, ny, nz).
            - t: Optional time at which the Dirichlet condition is applied.
        variable_index_map: VariableIndexMap object for mapping variable indices.
        mesh: UniformFVMesh object for accessing mesh properties.
        t: Optional time at which the Dirichlet condition is applied. May be None if
            the Dirichlet function does not depend on time.
        face_dim: Optional argument for when dirichlet_mode is "face-nodes". If
            provided, specifies the face dimension for quadrature points. Can be one of
            "x", "y", or "z".
        face_pos: Optional argument for when dirichlet_mode is "face-nodes". If
            provided, specifies the face position for quadrature points. Can be one of
            "l" (left) or "r" (right).
        p: Optional argument for when dirichlet_mode is "fv-averages" or "face-nodes".
            Specifies the polynomial degree of the quadrature rule used to evaluate the
            Dirichlet function.
    """
    st = slab_thickness
    slab_dim = AXIS_TO_DIM[axis]
    slab_pos = "l" if left else "r"
    slab_region = slab_dim + slab_pos
    outer_slice = crop(axis, (None, st) if left else (-st, None), ndim=4)

    if dirichlet_mode in {"fv-averages", "face-nodes"} and p is None:
        raise ValueError(
            f"Quadrature degree `p` must be provided for mode '{dirichlet_mode}'."
        )

    if dirichlet_mode == "fv-averages":
        if _u_.ndim != 4:
            raise ValueError(
                "For 'fv-averages' mode, _u_ must be 4D (nvars, nx, ny, nz)."
            )
        f_eval = mesh.perform_GaussLegendre_quadrature(
            lambda X, Y, Z: f(variable_index_map, X, Y, Z, t),
            node_axis=4,
            mesh_region=cast(Literal["xl", "xr", "yl", "yr", "zl", "zr"], slab_region),
            cell_region="interior",
            p=cast(int, p),
        )
        _u_[outer_slice] = f_eval
        return
    elif dirichlet_mode == "cell-centers":
        if _u_.ndim != 5 or _u_.shape[-1] != 1:
            raise ValueError(
                "For 'cell-centers' mode, _u_ must be 5D (nvars, nx, ny, nz, 1)."
            )
        X, Y, Z = mesh.get_cell_centers(
            cast(Literal["xl", "xr", "yl", "yr", "zl", "zr"], slab_region)
        )
        f_eval = f(variable_index_map, X, Y, Z, t)
        _u_[outer_slice + (0,)] = f_eval
        return
    elif dirichlet_mode == "face-nodes":
        if _u_.ndim != 5:
            raise ValueError(
                "For 'face-nodes' mode, _u_ must be 5D "
                "(nvars, nx, ny, nz, n_quadrature_points)."
            )
        if face_dim is None or face_pos is None:
            raise ValueError(
                "For 'face-nodes' mode, face_dim and face_pos must be provided."
            )
        cell_region = face_dim + face_pos
        X, Y, Z, _ = mesh.get_GaussLegendre_quadrature(
            mesh_region=cast(Literal["xl", "xr", "yl", "yr", "zl", "zr"], slab_region),
            cell_region=cast(Literal["xl", "xr", "yl", "yr", "zl", "zr"], cell_region),
            p=cast(int, p),
        )
        f_eval = f(variable_index_map, X, Y, Z, t)
        _u_[outer_slice + (slice(None),)] = f_eval
        return
    raise ValueError(
        f"Unsupported Dirichlet mode: {dirichlet_mode}. Supported modes are "
        "'fv-averages', 'cell-centers', and 'face-nodes'."
    )


def apply_free_bc(_u_: ArrayLike, slab_thickness: int, axis: int, left: bool):
    """
    Apply free boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        slab_thickness: Thickness of the slab (number of cells) to which the BC is
            applied.
        axis: Axis along which the BC is applied (1: x, 2: y, 3: z).
        left: Whether the BC is applied to the left (True) or right (False) boundary.
    """
    st = slab_thickness
    if left:
        outer_slice = crop(axis, (None, st))
        inner_slice = crop(axis, (st, st + 1))
    else:
        outer_slice = crop(axis, (-st, None))
        inner_slice = crop(axis, (-st - 1, -st))
    _u_[outer_slice] = _u_[inner_slice]


def apply_symmetric_bc(_u_: ArrayLike, slab_thickness: int, axis: int, left: bool):
    """
    Apply symmetric boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        slab_thickness: Thickness of the slab (number of cells) to which the BC is
            applied.
        axis: Axis along which the BC is applied (1: x, 2: y, 3: z).
        left: Whether the BC is applied to the left (True) or right (False) boundary.
    """
    st = slab_thickness
    flipper_slice = crop(axis, (None, None), step=-1)
    if left:
        outer_slice = crop(axis, (None, st))
        inner_slice = crop(axis, (st, 2 * st))
    else:
        outer_slice = crop(axis, (-st, None))
        inner_slice = crop(axis, (-2 * st, -st))
    _u_[outer_slice] = _u_[inner_slice][flipper_slice]


def apply_reflective_bc(
    _u_: ArrayLike,
    slab_thickness: int,
    axis: int,
    left: bool,
    variable_index_map: VariableIndexMap,
):
    """
    Apply reflective boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        slab_thickness: Thickness of the slab (number of cells) to which the BC is
            applied.
        axis: Axis along which the BC is applied (1: x, 2: y, 3: z).
        left: Whether the BC is applied to the left (True) or right (False) boundary.
        variable_index_map: VariableIndexMap object for mapping variable indices.
    """
    st = slab_thickness
    outer_slice = crop(axis, (None, st) if left else (-st, None))
    dim = AXIS_TO_DIM[axis]

    velocity = "v" + dim
    if velocity not in variable_index_map.var_names:
        raise ValueError(
            "VariableIndexMap must contain 'v' variable for reflective boundary conditions."
        )

    apply_symmetric_bc(_u_, slab_thickness, axis, left)
    _u_[outer_slice][variable_index_map(velocity)] *= -1


def apply_uniform_bc(
    _u_: ArrayLike, slab_thickness: int, axis: int, left: bool, value: float
):
    """
    Apply uniform boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        slab_thickness: Thickness of the slab (number of cells) to which the BC is
            applied.
        axis: Axis along which the BC is applied (1: x, 2: y, 3: z).
        left: Whether the BC is applied to the left (True) or right (False) boundary.
        value: Uniform value to apply at the boundary.
    """
    st = slab_thickness
    outer_slice = crop(axis, (None, st) if left else (-st, None))
    _u_[outer_slice] = value
