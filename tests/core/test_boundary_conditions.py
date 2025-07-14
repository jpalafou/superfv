import numpy as np
import pytest

from superfv.boundary_conditions import apply_bc
from superfv.mesh import UniformFVMesh
from superfv.tools.norms import linf_norm
from superfv.tools.slicing import VariableIndexMap, crop


@pytest.mark.parametrize("bc_type", ["periodic", "free", "symmetric", "zeros", "ones"])
def test_boundary_conditions_numpy_pad_equivalence(bc_type):
    """
    Test that the custom boundary condition application function produces the same
    result as the numpy pad function for various boundary condition types.
    """
    N = 32
    n_pad = 16

    u = np.random.rand(5, N, N, N)
    _u_ = np.empty((5, N + 2 * n_pad, N + 2 * n_pad, N + 2 * n_pad))
    _u_[:, n_pad:-n_pad, n_pad:-n_pad, n_pad:-n_pad] = u

    # baseline case: apply boundary conditions with np pad
    pad_kwargs = {
        "periodic": {"mode": "wrap"},
        "free": {"mode": "edge"},
        "symmetric": {"mode": "symmetric"},
        "zeros": {"mode": "constant", "constant_values": 0},
        "ones": {"mode": "constant", "constant_values": 1},
    }
    np_padded = np.pad(u, [(0,), (n_pad,), (n_pad,), (n_pad,)], **(pad_kwargs[bc_type]))

    # test case: apply boundary conditions with custom function
    apply_bc(
        _u_,
        (n_pad, n_pad, n_pad),
        mode=((bc_type, bc_type), (bc_type, bc_type), (bc_type, bc_type)),
    )

    assert np.array_equal(_u_, np_padded)


@pytest.mark.parametrize("ref_slab", ["xl", "xr", "yl", "yr", "zl", "zr"])
def test_reflective_boundary_conditions(ref_slab):
    """
    Test the application of reflective boundary conditions in a 3D array by setting one
    slab to be reflective and checking that the corresponding velocity component is
    negated, while the density and pressure remain unchanged.
    """
    N = 32
    n_pad = 16

    idx = VariableIndexMap({"rho": 0, "vx": 1, "vy": 2, "vz": 3, "P": 4})

    u = np.random.rand(5, N, N, N)
    _u_ref = np.empty((5, N + 2 * n_pad, N + 2 * n_pad, N + 2 * n_pad))
    _u_ref[:, n_pad:-n_pad, n_pad:-n_pad, n_pad:-n_pad] = u

    _u_sym = _u_ref.copy()

    apply_bc(
        _u_ref,
        (n_pad, n_pad, n_pad),
        mode=(
            (
                "reflective" if ref_slab == "xl" else "free",
                "reflective" if ref_slab == "xr" else "free",
            ),
            (
                "reflective" if ref_slab == "yl" else "free",
                "reflective" if ref_slab == "yr" else "free",
            ),
            (
                "reflective" if ref_slab == "zl" else "free",
                "reflective" if ref_slab == "zr" else "free",
            ),
        ),
        variable_index_map=idx,
    )

    apply_bc(
        _u_sym,
        (n_pad, n_pad, n_pad),
        mode=(
            (
                "symmetric" if ref_slab == "xl" else "free",
                "symmetric" if ref_slab == "xr" else "free",
            ),
            (
                "symmetric" if ref_slab == "yl" else "free",
                "symmetric" if ref_slab == "yr" else "free",
            ),
            (
                "symmetric" if ref_slab == "zl" else "free",
                "symmetric" if ref_slab == "zr" else "free",
            ),
        ),
    )

    ref_dim, ref_pos = ref_slab[0], ref_slab[1]
    DIM_TO_AXIS = {"x": 1, "y": 2, "z": 3}

    assert np.array_equal(_u_ref[idx("rho")], _u_sym[idx("rho")])
    assert np.array_equal(_u_ref[idx("P")], _u_sym[idx("P")])
    assert np.array_equal(
        _u_ref[idx("v" + ref_dim, keepdims=True)][
            crop(
                DIM_TO_AXIS[ref_dim],
                (None, n_pad) if ref_pos == "l" else (-n_pad, None),
            )
        ],
        -_u_sym[idx("v" + ref_dim, keepdims=True)][
            crop(
                DIM_TO_AXIS[ref_dim],
                (None, n_pad) if ref_pos == "l" else (-n_pad, None),
            )
        ],
    )


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_dirichlet_boundary_condition_fv_averages(dims, p):
    """
    Test the application of a Dirichlet boundary condition using the FV averages
    method.
    """
    N = 32
    n_pad = 16

    idx = VariableIndexMap({"rho": 0, "vx": 1, "vy": 2, "vz": 3, "P": 4})
    t = 0.1

    # define a sinusoidal function for the boundary condition
    def sinus(idx, x, y, z, t):
        r = np.zeros(x.shape)
        u = np.zeros((5, *x.shape))
        if "x" in dims:
            r += x
        if "y" in dims:
            r += y
        if "z" in dims:
            r += z
        u[idx("rho")] = t * np.sin(2 * np.pi * r)
        u[idx("vx")] = 2 * t * np.sin(2 * np.pi * r)
        u[idx("vy")] = 2 * t * np.sin(2 * np.pi * r)
        u[idx("vz")] = 2 * t * np.sin(2 * np.pi * r)
        u[idx("P")] = 10 * t * np.sin(2 * np.pi * r)
        return u

    # create meshes
    mesh = UniformFVMesh(
        nx=N if "x" in dims else 1,
        ny=N if "y" in dims else 1,
        nz=N if "z" in dims else 1,
        slab_depth=n_pad,
    )
    megamesh = UniformFVMesh(
        nx=N + 2 * n_pad if "x" in dims else 1,
        ny=N + 2 * n_pad if "y" in dims else 1,
        nz=N + 2 * n_pad if "z" in dims else 1,
        xlim=(-0.5, 1.5) if "x" in dims else (0, 1),
        ylim=(-0.5, 1.5) if "y" in dims else (0, 1),
        zlim=(-0.5, 1.5) if "z" in dims else (0, 1),
        slab_depth=0,
    )
    assert mesh._shape_ == megamesh.shape

    # baseline case: apply dirichlet boundary function to megamesh
    megamesh_u = megamesh.perform_GaussLegendre_quadrature(
        lambda X, Y, Z: sinus(idx, X, Y, Z, t),
        4,
        mesh_region="core",
        cell_region="interior",
        p=p,
    )

    # test case: apply dirichlet boundary condition with custom function
    u = mesh.perform_GaussLegendre_quadrature(
        lambda X, Y, Z: sinus(idx, X, Y, Z, t),
        4,
        mesh_region="core",
        cell_region="interior",
        p=p,
    )
    _u_ = np.empty(
        (
            5,
            N + 2 * n_pad if "x" in dims else 1,
            N + 2 * n_pad if "y" in dims else 1,
            N + 2 * n_pad if "z" in dims else 1,
        )
    )
    _u_[
        :,
        slice(n_pad, -n_pad) if "x" in dims else slice(1),
        slice(n_pad, -n_pad) if "y" in dims else slice(1),
        slice(n_pad, -n_pad) if "z" in dims else slice(1),
    ] = u
    apply_bc(
        _u_,
        (
            n_pad if "x" in dims else 0,
            n_pad if "y" in dims else 0,
            n_pad if "z" in dims else 0,
        ),
        mode=(
            ("dirichlet", "dirichlet") if "x" in dims else ("none", "none"),
            ("dirichlet", "dirichlet") if "y" in dims else ("none", "none"),
            ("dirichlet", "dirichlet") if "z" in dims else ("none", "none"),
        ),
        f=((sinus, sinus), (sinus, sinus), (sinus, sinus)),
        dirichlet_mode="fv-averages",
        mesh=mesh,
        variable_index_map=idx,
        t=t,
        p=p,
    )

    assert linf_norm(_u_ - megamesh_u) < 1e-15


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_dirichlet_boundary_condition_cell_centers(dims):
    """
    Test the application of a Dirichlet boundary condition using the cell centers
    method.
    """
    N = 32
    n_pad = 16

    idx = VariableIndexMap({"rho": 0, "vx": 1, "vy": 2, "vz": 3, "P": 4})
    t = 0.1

    # define a sinusoidal function for the boundary condition
    def sinus(idx, x, y, z, t):
        r = np.zeros(x.shape)
        u = np.zeros((5, *x.shape))
        if "x" in dims:
            r += x
        if "y" in dims:
            r += y
        if "z" in dims:
            r += z
        u[idx("rho")] = t * np.sin(2 * np.pi * r)
        u[idx("vx")] = 2 * t * np.sin(2 * np.pi * r)
        u[idx("vy")] = 2 * t * np.sin(2 * np.pi * r)
        u[idx("vz")] = 2 * t * np.sin(2 * np.pi * r)
        u[idx("P")] = 10 * t * np.sin(2 * np.pi * r)
        return u

    # create meshes
    mesh = UniformFVMesh(
        nx=N if "x" in dims else 1,
        ny=N if "y" in dims else 1,
        nz=N if "z" in dims else 1,
        slab_depth=n_pad,
    )
    megamesh = UniformFVMesh(
        nx=N + 2 * n_pad if "x" in dims else 1,
        ny=N + 2 * n_pad if "y" in dims else 1,
        nz=N + 2 * n_pad if "z" in dims else 1,
        xlim=(-0.5, 1.5) if "x" in dims else (0, 1),
        ylim=(-0.5, 1.5) if "y" in dims else (0, 1),
        zlim=(-0.5, 1.5) if "z" in dims else (0, 1),
        slab_depth=0,
    )
    assert mesh._shape_ == megamesh.shape

    # baseline case: apply dirichlet boundary function to megamesh
    X, Y, Z = megamesh.get_cell_centers()
    megamesh_u = sinus(idx, X, Y, Z, t)[:, :, :, :, np.newaxis]

    # test case: apply dirichlet boundary condition with custom function
    X, Y, Z = mesh.get_cell_centers()
    u = sinus(idx, X, Y, Z, t)[:, :, :, :, np.newaxis]
    _u_ = np.empty(
        (
            5,
            N + 2 * n_pad if "x" in dims else 1,
            N + 2 * n_pad if "y" in dims else 1,
            N + 2 * n_pad if "z" in dims else 1,
            1,
        )
    )
    _u_[
        :,
        slice(n_pad, -n_pad) if "x" in dims else slice(1),
        slice(n_pad, -n_pad) if "y" in dims else slice(1),
        slice(n_pad, -n_pad) if "z" in dims else slice(1),
        :,
    ] = u
    apply_bc(
        _u_,
        (
            n_pad if "x" in dims else 0,
            n_pad if "y" in dims else 0,
            n_pad if "z" in dims else 0,
        ),
        mode=(
            ("dirichlet", "dirichlet") if "x" in dims else ("none", "none"),
            ("dirichlet", "dirichlet") if "y" in dims else ("none", "none"),
            ("dirichlet", "dirichlet") if "z" in dims else ("none", "none"),
        ),
        f=((sinus, sinus), (sinus, sinus), (sinus, sinus)),
        dirichlet_mode="cell-centers",
        mesh=mesh,
        variable_index_map=idx,
        t=t,
    )

    assert np.array_equal(_u_, megamesh_u)


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("p", [0, 1, 2, 3])
@pytest.mark.parametrize("face_pos", ["l", "r"])
def test_dirichlet_boundary_condition_face_nodes(dims, p, face_pos):
    """
    Test the application of a Dirichlet boundary condition using the face nodes
    method.
    """
    N = 32
    n_pad = 16
    cell_region = dims[0] + face_pos

    idx = VariableIndexMap({"rho": 0, "vx": 1, "vy": 2, "vz": 3, "P": 4})
    t = 0.1

    # define a sinusoidal function for the boundary condition
    def sinus(idx, x, y, z, t):
        r = np.zeros(x.shape)
        u = np.zeros((5, *x.shape))
        if "x" in dims:
            r += x
        if "y" in dims:
            r += y
        if "z" in dims:
            r += z
        u[idx("rho")] = t * np.sin(2 * np.pi * r)
        u[idx("vx")] = 2 * t * np.sin(2 * np.pi * r)
        u[idx("vy")] = 2 * t * np.sin(2 * np.pi * r)
        u[idx("vz")] = 2 * t * np.sin(2 * np.pi * r)
        u[idx("P")] = 10 * t * np.sin(2 * np.pi * r)
        return u

    # create meshes
    mesh = UniformFVMesh(
        nx=N if "x" in dims else 1,
        ny=N if "y" in dims else 1,
        nz=N if "z" in dims else 1,
        slab_depth=n_pad,
    )
    megamesh = UniformFVMesh(
        nx=N + 2 * n_pad if "x" in dims else 1,
        ny=N + 2 * n_pad if "y" in dims else 1,
        nz=N + 2 * n_pad if "z" in dims else 1,
        xlim=(-0.5, 1.5) if "x" in dims else (0, 1),
        ylim=(-0.5, 1.5) if "y" in dims else (0, 1),
        zlim=(-0.5, 1.5) if "z" in dims else (0, 1),
        slab_depth=0,
    )
    assert mesh._shape_ == megamesh.shape

    # baseline case: apply dirichlet boundary function to megamesh
    X, Y, Z, _ = megamesh.get_GaussLegendre_quadrature("core", cell_region, p)
    megamesh_u = sinus(idx, X, Y, Z, t)

    # test case: apply dirichlet boundary condition with custom function
    X, Y, Z, _ = mesh.get_GaussLegendre_quadrature("core", cell_region, p)
    u = sinus(idx, X, Y, Z, t)
    _u_ = np.empty(
        (
            5,
            N + 2 * n_pad if "x" in dims else 1,
            N + 2 * n_pad if "y" in dims else 1,
            N + 2 * n_pad if "z" in dims else 1,
            u.shape[4],
        )
    )
    _u_[
        :,
        slice(n_pad, -n_pad) if "x" in dims else slice(1),
        slice(n_pad, -n_pad) if "y" in dims else slice(1),
        slice(n_pad, -n_pad) if "z" in dims else slice(1),
        :,
    ] = u
    apply_bc(
        _u_,
        (
            n_pad if "x" in dims else 0,
            n_pad if "y" in dims else 0,
            n_pad if "z" in dims else 0,
        ),
        mode=(
            ("dirichlet", "dirichlet") if "x" in dims else ("none", "none"),
            ("dirichlet", "dirichlet") if "y" in dims else ("none", "none"),
            ("dirichlet", "dirichlet") if "z" in dims else ("none", "none"),
        ),
        f=((sinus, sinus), (sinus, sinus), (sinus, sinus)),
        dirichlet_mode="face-nodes",
        mesh=mesh,
        variable_index_map=idx,
        t=t,
        p=p,
        face_dim=cell_region[0],
        face_pos=cell_region[1],
    )

    assert linf_norm(_u_ - megamesh_u) == 0
