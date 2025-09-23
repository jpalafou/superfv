from dataclasses import dataclass
from types import ModuleType
from typing import Literal, Optional, Tuple

from superfv.fv import DIM_TO_AXIS
from superfv.interpolation_schemes import InterpolationScheme, LimiterConfig
from superfv.slope_limiting.smooth_extrema_detection import smooth_extrema_detector
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop, modify_slices


@dataclass(frozen=True, slots=True)
class musclConfig(LimiterConfig):
    """
    Configuration for the MUSCL slope limiter.

    Attributes:
        limiter: The type of limiter to use. Can be "minmod", "moncen", or None.
        SED: Whether to use the smooth extrema detector to relax the limiter.
    """

    limiter: Optional[Literal["minmod", "moncen"]]
    SED: bool

    def key(self) -> str:
        return f"muscl-{self.limiter}"

    def to_dict(self) -> dict:
        return dict(
            limiter=self.limiter,
            SED=self.SED,
        )


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
    limiter_config: musclConfig = musclConfig(limiter="minmod", SED=False)

    def __post_init__(self):
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
    limiter: Optional[Literal["minmod", "moncen"]] = None,
    SED: bool = False,
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
        buffer: Array to store intermediate results. Has shape
            (nvars, nx, ny, nz, >=10) for 1D,
            (nvars, nx, ny, nz, >=12) for 2D,
            or (nvars, nx, ny, nz, >=13) for 3D
        limiter: Limiter to apply to the slopes. Can be "minmod" or "moncen".

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    # define slices for left, center, and right nodes
    slc_l = crop(DIM_TO_AXIS[face_dim], (None, -2), ndim=4)
    slc_c = crop(DIM_TO_AXIS[face_dim], (1, -1), ndim=4)
    slc_r = crop(DIM_TO_AXIS[face_dim], (2, None), ndim=4)
    buff_slc = crop(DIM_TO_AXIS[face_dim], (1, -1), ndim=5)
    out_slc = modify_slices(buff_slc, 4, 0)

    # allocate arrays
    dlft = buffer[modify_slices(buff_slc, 4, 0)]
    drgt = buffer[modify_slices(buff_slc, 4, 1)]
    dcen = buffer[modify_slices(buff_slc, 4, 2)]
    dsgn = buffer[modify_slices(buff_slc, 4, 3)]
    dslp = buffer[modify_slices(buff_slc, 4, 4)]
    alpha = buffer[..., 5:6]
    abuff = buffer[..., 6:]

    # compute smooth extrema detector if requested
    if SED:
        out_modified = smooth_extrema_detector(
            xp, u, active_dims, out=alpha, buffer=abuff
        )
    else:
        out_modified = modify_slices(buff_slc, 4, slice(0, 1))

    # write slopes to `out` array
    if limiter == "minmod":
        dlft[...] = u[slc_c] - u[slc_l]
        drgt[...] = u[slc_r] - u[slc_c]
        dcen[...] = 0.5 * (dlft + drgt)
        dsgn[...] = xp.sign(dlft)
        dslp[...] = dsgn * xp.minimum(xp.abs(dlft), xp.abs(drgt))
        out[out_slc] = xp.where(dlft * drgt <= 0, 0, dslp)
        if SED:
            out[out_slc] = xp.where(alpha[out_slc] < 1, out[out_slc], dcen)
    elif limiter == "moncen":
        dlft[...] = u[slc_c] - u[slc_l]
        drgt[...] = u[slc_r] - u[slc_c]
        dcen[...] = 0.5 * (dlft + drgt)
        dsgn[...] = xp.sign(dcen)
        dslp[...] = dsgn * xp.minimum(
            xp.minimum(xp.abs(2 * dlft), 2 * xp.abs(drgt)), xp.abs(dcen)
        )
        out[out_slc] = xp.where(dlft * drgt <= 0, 0, dslp)
        if SED:
            out[out_slc] = xp.where(alpha[out_slc] < 1, out[out_slc], dcen)
    elif limiter is None:
        out[out_slc] = 0.5 * (u[slc_r] - u[slc_l])
    else:
        raise ValueError(f"Unknown limiter: {limiter}.")

    return out_modified
