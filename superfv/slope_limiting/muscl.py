from dataclasses import dataclass
from types import ModuleType
from typing import Literal, Optional, Tuple, cast

from superfv.fv import DIM_TO_AXIS
from superfv.interpolation_schemes import InterpolationScheme, LimiterConfig
from superfv.slope_limiting import gather_neighbor_slices
from superfv.slope_limiting.smooth_extrema_detection import smooth_extrema_detector
from superfv.tools.buffer import check_buffer_slots
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop, insert_slice, merge_slices, replace_slice


@dataclass(frozen=True, slots=True)
class musclConfig(LimiterConfig):
    """
    Configuration for the MUSCL slope limiter.

    Attributes:
        shock_detection: Whether to enable shock detection.
        smooth_extrema_detection: Whether to enable smooth extrema detection.
        physical_admissibility_detection: Whether to enable physical admissibility
            detection (PAD).
        eta_max: Eta threshold for shock detection if shock_detection is True.
        PAD_bounds: Array with shape (nvars, 2) specifying the lower and upper bounds,
            respectively, for each variable when physical_admissibility_detection is
            True. Must be provided if physical_admissibility_detection is True.
        PAD_atol: Absolute tolerance for physical admissibility detection if
            physical_admissibility_detection is True.
        limiter: Optional slope limiter specification of "minmod", "moncen", "PP2D".
    """

    limiter: Optional[Literal["minmod", "moncen", "PP2D"]] = None

    def key(self) -> str:
        return f"muscl-{self.limiter}"

    def to_dict(self) -> dict:
        out = LimiterConfig.to_dict(self)
        out.update(dict(limiter=self.limiter))
        return out


@dataclass(frozen=True, slots=True)
class musclInterpolationScheme(InterpolationScheme):
    """
    Configuration for MUSCL interpolation schemes.

    Attributes:
        p: The polynomial degree. Must be 1.
        flux_recipe: The flux recipe to use. For MUSCL schemes, this simplifies to:
            - 1: compute conservative slopes -> limit conservative slopes -> compute
                primitive nodes -> compute fluxes
            - 2: compute primitive cell averages -> compute primitive slopes -> limit
                primitive slopes -> compute fluxes
        limiter_config: The MUSCL limiter configuration to use.
    """

    p: int = 1
    flux_recipe: Literal[1, 2] = 2
    limiter_config: musclConfig = musclConfig(
        shock_detection=False,
        smooth_extrema_detection=False,
        physical_admissibility_detection=False,
    )

    def __post_init__(self):
        InterpolationScheme.__post_init__(self)
        if self.p != 1:
            raise ValueError("musclInterpolationScheme must have p=1")
        if not isinstance(self.limiter_config, musclConfig):
            raise ValueError("musclInterpolationScheme requires a musclConfig")

    def key(self) -> str:
        return self.limiter_config.key()

    def to_dict(self) -> dict:
        return dict(
            p=self.p,
            flux_recipe=self.flux_recipe,
            limiter_config=(
                None if self.limiter_config is None else self.limiter_config.to_dict()
            ),
        )


def compute_limited_slopes(
    xp: ModuleType,
    u: ArrayLike,
    face_dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    limiter: Optional[Literal["minmod", "moncen", "PP2D"]] = None,
    SED: bool = False,
    alpha: Optional[ArrayLike] = None,
) -> Tuple[slice, ...]:
    """
    Compute limited slopes for face-centered nodes from an array of finite
    volume averages.

    Args:
        xp: `np` namespace.
        u: Array of finite volume averages to compute slopes from, has shape
            (nvars, nx, ny, nz).
        face_dim: Dimension along which the limited slopes are computed.
        active_dims: Tuple indicating the active dimensions for interpolation. Can be
            some combination of 'x', 'y', and 'z'. For example, ('x', 'y') for the
            interpolation of nodes along the face of a cell on a two-dimensional grid.
        out: Output array to store the limited slopes. Has shape
            (nvars, nx, ny, nz, nout). The result is stored in out[..., 0].
        buffer: Array to which temporary values are assigned. Has different shape
            requirements depending on whether SED is used and the number (length) of
            active dimensions:
            - without SED: (nvars, nx, ny, nz, >=5)
            - with SED, 1D: (nvars, nx, ny, nz, >=12)
            - with SED, 2D: (nvars, nx, ny, nz, >=14)
            - with SED, 3D: (nvars, nx, ny, nz, >=15)
        limiter: Limiter to apply to the slopes. Can be "minmod" or "moncen".
        SED: Whether to use the smooth extrema detector to relax the limiter.
        alpha: Array to store the smooth extrema detector values if SED is True. Has
            shape (nvars, nx, ny, nz, 1).

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    # define slices for left, center, and right nodes
    slc_l = crop(DIM_TO_AXIS[face_dim], (None, -2), ndim=4)
    slc_c = crop(DIM_TO_AXIS[face_dim], (1, -1), ndim=4)
    slc_r = crop(DIM_TO_AXIS[face_dim], (2, None), ndim=4)
    inner = insert_slice(slc_c, 4, 0)

    # allocate arrays
    check_buffer_slots(buffer, required=5)
    dlft = buffer[replace_slice(inner, 4, 0)]
    drgt = buffer[replace_slice(inner, 4, 1)]
    dcen = buffer[replace_slice(inner, 4, 2)]
    dsgn = buffer[replace_slice(inner, 4, 3)]
    dslp = buffer[replace_slice(inner, 4, 4)]

    # compute smooth extrema detector if requested
    if SED:
        if alpha is None:
            raise ValueError("alpha array must be provided when SED is True.")
        abuf = buffer[..., 5:]
        modified = smooth_extrema_detector(xp, u, active_dims, out=alpha, buffer=abuf)
    else:
        modified = cast(Tuple[slice, ...], replace_slice(inner, 4, slice(None, 1)))

    # write slopes to `out` array
    match limiter:
        case "minmod":
            dlft[...] = u[slc_c] - u[slc_l]
            drgt[...] = u[slc_r] - u[slc_c]
            dcen[...] = 0.5 * (dlft + drgt)
            dsgn[...] = xp.sign(dlft)
            dslp[...] = dsgn * xp.minimum(xp.abs(dlft), xp.abs(drgt))
            out[inner] = xp.where(dlft * drgt <= 0, 0, dslp)
            if SED:
                out[inner] = xp.where(
                    cast(ArrayLike, alpha)[inner] < 1, out[inner], dcen
                )
        case "moncen":
            dlft[...] = u[slc_c] - u[slc_l]
            drgt[...] = u[slc_r] - u[slc_c]
            dcen[...] = 0.5 * (dlft + drgt)
            dsgn[...] = xp.sign(dcen)
            dslp[...] = dsgn * xp.minimum(
                xp.minimum(xp.abs(2 * dlft), 2 * xp.abs(drgt)), xp.abs(dcen)
            )
            out[inner] = xp.where(dlft * drgt <= 0, 0, dslp)
            if SED:
                out[inner] = xp.where(
                    cast(ArrayLike, alpha)[inner] < 1, out[inner], dcen
                )
        case "PP2D":
            raise ValueError("Oops, use the `compute_PP2D_slopes` function instead.")
        case None:
            out[inner] = 0.5 * (u[slc_r] - u[slc_l])
        case _:
            raise ValueError(f"Unknown limiter: {limiter}.")

    return modified


def compute_PP2D_slopes(
    xp: ModuleType,
    u: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    Sx: ArrayLike,
    Sy: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-20,
    SED: bool = False,
    alpha: Optional[ArrayLike] = None,
) -> Tuple[slice, ...]:
    """
    Compute PP2D limited slopes and write them to the 'Sx' and 'Sy' arrays.

    Args:
        xp: `np` namespace.
        u: Array of finite volume averages to compute slopes from, has shape
            (nvars, nx, ny, nz).
        active_dims: Tuple indicating the active dimensions for interpolation. Can be
            some combination of 'x', 'y', and 'z'. For example, ('x', 'y') for the
            interpolation of nodes along the face of a cell on a two-dimensional grid.
        Sx: Output array to store the limited slopes in the first active dimension. Has
            shape (nvars, nx, ny, nz, 1).
        Sy: Output array to store the limited slopes in the second active dimension.
            Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which temporary values are assigned. Has different shape
            requirements depending on whether SED is used and the number (length) of
            active dimensions:
            - without SED: (nvars, nx, ny, nz, >=4)
            - with SED, 1D: (nvars, nx, ny, nz, >=11)
            - with SED, 2D: (nvars, nx, ny, nz, >=13)
            - with SED, 3D: (nvars, nx, ny, nz, >=14)
        eps: Small number to avoid division by zero.
        SED: Whether to use the smooth extrema detector to relax the limiter.
        alpha: Array to store the smooth extrema detector values if SED is True. Has
            shape (nvars, nx, ny, nz, 1).

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    if len(active_dims) != 2:
        raise ValueError("PP2D slope limiter requires exactly two active dimensions.")

    # allocate arrays
    check_buffer_slots(buffer, required=4)
    V_min = buffer[..., 0]
    V_max = buffer[..., 1]
    V = buffer[..., 2]
    theta = buffer[..., 3:4]

    V_min_neighbors = xp.empty((8,) + u.shape)
    V_max_neighbors = xp.empty((8,) + u.shape)

    # compute second-order slopes
    axis1 = DIM_TO_AXIS[active_dims[0]]
    axis2 = DIM_TO_AXIS[active_dims[1]]
    slc1_l = crop(axis1, (None, -2), ndim=4)
    slc1_r = crop(axis1, (2, None), ndim=4)
    slc2_l = crop(axis2, (None, -2), ndim=4)
    slc2_r = crop(axis2, (2, None), ndim=4)

    slc1_c = replace_slice(crop(axis1, (1, -1), ndim=5), 4, slice(None, 1))
    slc2_c = replace_slice(crop(axis2, (1, -1), ndim=5), 4, slice(None, 1))

    Sx[replace_slice(slc1_c, 4, 0)] = 0.5 * (u[slc1_r] - u[slc1_l])
    Sy[replace_slice(slc2_c, 4, 0)] = 0.5 * (u[slc2_r] - u[slc2_l])

    # compute PPD2 limiter
    neighbor_slices = gather_neighbor_slices(active_dims, include_corners=True)
    c_slc = neighbor_slices[0]

    V_min_neighbors[insert_slice(c_slc, 0, slice(None))] = xp.array(
        [u[slc] - u[c_slc] for slc in neighbor_slices[1:]]
    )
    V_min[...] = xp.minimum(xp.min(V_min_neighbors, axis=0), -eps)

    V_max_neighbors[insert_slice(c_slc, 0, slice(None))] = xp.array(
        [u[slc] - u[c_slc] for slc in neighbor_slices[1:]]
    )
    V_max[...] = xp.maximum(xp.max(V_max_neighbors, axis=0), eps)

    V[...] = (
        2
        * xp.minimum(xp.abs(V_min), xp.abs(V_max))
        / (xp.abs(Sx[..., 0]) + xp.abs(Sy[..., 0]))
    )
    theta[..., 0] = xp.minimum(V, 1)

    # apply SED if requested
    if SED:
        if alpha is None:
            raise ValueError("alpha array must be provided when SED is True.")
        abuff = buffer[..., 4:]
        modified = smooth_extrema_detector(xp, u, active_dims, out=alpha, buffer=abuff)
        theta[...] = xp.where(alpha < 1, theta, 1.0)
    else:
        modified = merge_slices(
            cast(Tuple[slice, ...], slc1_c), cast(Tuple[slice, ...], slc2_c)
        )

    Sx[modified] = theta[modified] * Sx[modified]
    Sy[modified] = theta[modified] * Sy[modified]

    return modified
