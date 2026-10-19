from functools import lru_cache
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.tools.slicing import crop, merge_slices
from superfv.tools.step_history import MultiTimer

from .axes import DIM_TO_AXIS, XYZ_TUPLE
from .configs import (
    BoundaryConditionParameters,
    FV_SchemeParameters,
    HydroParameters,
    ZhangShuParameters,
)
from .finite_volume_driver import (
    Stencil,
    _stencil_cache,
    apply_fv_bc,
    get_interior_view,
    integrate_cell_averages,
    integrate_gauss_legendre_face_nodes,
    integrate_transverse_nodes,
    interpolate_cell_centers,
    interpolate_face_nodes,
    interpolate_interface_nodes,
)
from .hydro import cons_to_prim, prim_to_cs
from .mesh import UniformFiniteVolumeMesh
from .riemann_solvers import solve_riemann_problem
from .slope_limiting import apply_global_bounds, compute_dmp
from .slope_limiting.muscl import compute_MUSCL_slopes, reconstruct_MUSCL_faces
from .slope_limiting.shock_detection import detect_shocks
from .slope_limiting.smooth_extrema_detection import compute_alpha
from .slope_limiting.zhang_and_shu import compute_theta
from .stencils import (
    conservative_interpolation,
    finite_difference,
    transverse_integration,
)
from .sweep import stencil_sweep
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.variable_index_map import VariableIndexMap

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

def update_advection_workspace(
    u: ArrayLike,
    _u_: ArrayLike,
    t: float,
    idx: VariableIndexMap,
    mesh: UniformFiniteVolumeMesh,
    bc_params: BoundaryConditionParameters,
    timer: Optional[MultiTimer] = None,
):
    """
    Update the `_u_` and array with ghost-cell-padded finite volume averages.

    u: shape (nvars, nx, ny, nz) - Input array of cell averages without ghost cells.
        Is not modified.
    _u_: shape (nvars, mesh._nx_, mesh._ny_, mesh._nz_) - Array to which u is written with boundary
        conditions applied to ghost cells along active dimensions.
    t: Current simulation time, used for time-dependent boundary conditions.
    idx: VariableIndexMap for indexing into the variable dimension of the arrays.
    mesh: Mesh object containing information about the mesh and its dimensions.
    bc_params: Parameters for the boundary conditions.
    timer: Optional[MultiTimer] = None - Timer object for performance measurement.
    """
    using_cupy = CUPY_AVAILABLE and isinstance(_u_, cp.ndarray)
    active_dims = mesh.active_dims
    interior = get_interior_view(active_dims, mesh.nghost)

    # Update interior conservative FV averages and apply BC
    timer is not None and timer.start("boundary_conditions", using_cupy)  # TIMER START
    _u_[interior] = u
    apply_fv_bc(_u_, idx, mesh, t, bc_params, None)
    timer is not None and timer.stop("boundary_conditions", using_cupy)  # TIMER STOP



