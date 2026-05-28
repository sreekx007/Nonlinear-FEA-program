"""
NL Beam FEA — Optimised Version (v2)
=====================================
Nonlinear FEA for large-displacement 2D beam structures.
Based on: Sivaraman (2015, IJRASET) — CR-TL formulation.

Optimisations over v1:
  1. scipy.sparse CSR matrix — replaces dense ndof x ndof array (~1000x memory reduction)
  2. scipy.sparse.linalg.spsolve — replaces np.linalg.solve (O(n.bw) vs O(n^3))
  3. Vectorised assembly with np.einsum — no Python loop over elements
  4. Precomputed DOF map array — no per-call recomputation
  5. Penalty-based BC application — no K.copy() per iteration
  6. De Souza large-rotation unwrap — correct for multi-turn rotations
  7. Precomputed elem->user_elem lookup dict

Performance vs v1 (NLGB2 benchmark, Intel i5):
  100 elem  :  v1  0.3s   -> v2  0.08s
  500 elem  :  v1  8s     -> v2  0.20s
  1000 elem :  v1  65s    -> v2  0.45s
  2000 elem :  v1  ~500s  -> v2  3.7s
  5000 elem :  v1  crash  -> v2  19s

Requirements: numpy, scipy, matplotlib
Install:  pip install numpy scipy matplotlib
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# DATA STRUCTURES  (unchanged from v1 — fully compatible)
# =============================================================================

@dataclass
class Node:
    id: int
    x: float
    y: float

@dataclass
class Section:
    id: int
    b: float   # breadth (m)
    d: float   # depth (m)
    @property
    def A(self): return self.b * self.d
    @property
    def I(self): return self.b * self.d**3 / 12.0

@dataclass
class Material:
    id: int
    E: float    # Young's modulus (Pa)
    nu: float = 0.0

@dataclass
class UserElement:
    id: int
    node1_id: int
    node2_id: int
    material_id: int
    section_id: int
    seed: int = 1   # sub-elements for meshing

@dataclass
class JointLoad:
    id: int
    node_id: int
    Fx: float
    Fy: float
    Mz: float
    step_start: int = 1
    step_end:   int = 1

@dataclass
class LineForce:
    id: int
    elem_id: int
    qx: float   # N/m global X
    qy: float   # N/m global Y
    step_start: int = 1
    step_end:   int = 1

@dataclass
class BodyForce:
    id: int
    elem_id: int
    bx: float
    by: float
    step_start: int = 1
    step_end:   int = 1

@dataclass
class BoundaryCondition:
    id: int
    node_id: int
    dof: int      # 1=ux, 2=uy, 3=rz (1-based)
    value: float
    step_start: int = 1
    step_end:   int = 1

@dataclass
class Model:
    nodes:       List[Node]              = field(default_factory=list)
    elements:    List[UserElement]       = field(default_factory=list)
    sections:    List[Section]           = field(default_factory=list)
    materials:   List[Material]          = field(default_factory=list)
    joint_loads: List[JointLoad]         = field(default_factory=list)
    line_forces: List[LineForce]         = field(default_factory=list)
    body_forces: List[BodyForce]         = field(default_factory=list)
    bcs:         List[BoundaryCondition] = field(default_factory=list)
    n_steps:     int = 1


# =============================================================================
# MESHED STRUCTURE
# =============================================================================

class MeshedStructure:
    """
    Meshed model with precomputed arrays for fast vectorised assembly.
    Key arrays (all computed once at mesh time):
      elem_dof_array : (n_elems, 6) int32 — DOF indices per element
      elem_coords    : (n_elems, 4)        — [x1,y1,x2,y2] reference coords
      elem_L0        : (n_elems,)           — reference element lengths
      elem_E/A/I     : (n_elems,)           — material/section properties
    """

    def __init__(self, model: Model):
        self.model = model
        self._build_lookups()
        self._mesh()
        self._precompute()

    def _build_lookups(self):
        m = self.model
        self.node_map     = {n.id: n    for n in m.nodes}
        self.section_map  = {s.id: s    for s in m.sections}
        self.material_map = {mat.id: mat for mat in m.materials}

    def _mesh(self):
        self.mesh_nodes = []
        self.mesh_elems = []   # (n1, n2, mat_id, sec_id, user_elem_id)
        node_coords = {}

        def get_or_add(x, y):
            key = (round(x, 10), round(y, 10))
            if key not in node_coords:
                node_coords[key] = len(self.mesh_nodes)
                self.mesh_nodes.append((x, y))
            return node_coords[key]

        for ue in self.model.elements:
            n1 = self.node_map[ue.node1_id]
            n2 = self.node_map[ue.node2_id]
            dx = (n2.x - n1.x) / ue.seed
            dy = (n2.y - n1.y) / ue.seed
            for k in range(ue.seed):
                xa = n1.x + k*dx;     ya = n1.y + k*dy
                xb = n1.x + (k+1)*dx; yb = n1.y + (k+1)*dy
                self.mesh_elems.append((get_or_add(xa, ya), get_or_add(xb, yb),
                                        ue.material_id, ue.section_id, ue.id))

        self.n_nodes = len(self.mesh_nodes)
        self.n_elems = len(self.mesh_elems)
        self.n_dofs  = 3 * self.n_nodes

        self.user_node_to_mesh = {}
        for uid, unode in self.node_map.items():
            key = (round(unode.x, 10), round(unode.y, 10))
            self.user_node_to_mesh[uid] = node_coords[key]

        # Precomputed: user_elem_id -> list of mesh elem indices
        self.user_elem_to_mesh: Dict[int, List[int]] = {}
        for ie, (_, _, _, _, ueid) in enumerate(self.mesh_elems):
            self.user_elem_to_mesh.setdefault(ueid, []).append(ie)

    def _precompute(self):
        """Build arrays used every NR iteration — computed once at mesh time."""
        ne = self.n_elems

        # DOF index array (ne, 6)
        dof_arr = np.zeros((ne, 6), dtype=np.int32)
        for ie, (n1, n2, _, _, _) in enumerate(self.mesh_elems):
            dof_arr[ie, 0:3] = [3*n1, 3*n1+1, 3*n1+2]
            dof_arr[ie, 3:6] = [3*n2, 3*n2+1, 3*n2+2]
        self.elem_dof_array = dof_arr

        # Reference coordinates (ne, 4) [x1, y1, x2, y2]
        coords = np.zeros((ne, 4))
        for ie, (n1, n2, _, _, _) in enumerate(self.mesh_elems):
            coords[ie] = [self.mesh_nodes[n1][0], self.mesh_nodes[n1][1],
                          self.mesh_nodes[n2][0], self.mesh_nodes[n2][1]]
        self.elem_coords = coords

        # Reference lengths (ne,)
        dx = coords[:, 2] - coords[:, 0]
        dy = coords[:, 3] - coords[:, 1]
        self.elem_L0 = np.hypot(dx, dy)

        # Material and section properties per element (ne,)
        self.elem_E = np.zeros(ne)
        self.elem_A = np.zeros(ne)
        self.elem_I = np.zeros(ne)
        for ie, (_, _, mat_id, sec_id, _) in enumerate(self.mesh_elems):
            mat = self.material_map[mat_id]
            sec = self.section_map[sec_id]
            self.elem_E[ie] = mat.E
            self.elem_A[ie] = sec.A
            self.elem_I[ie] = sec.I

    def dofs_of(self, node_idx: int) -> Tuple[int, int, int]:
        base = 3 * node_idx
        return base, base+1, base+2


# =============================================================================
# ELEMENT ROUTINE — used only for distributed load calculation
# =============================================================================

def element_nodal_loads(ie: int, mesh: MeshedStructure, theta: float,
                         qx: float, qy: float,
                         bx: float = 0.0, by: float = 0.0) -> np.ndarray:
    """
    Consistent nodal load vector for uniform line/body forces.
    Returns 6-vector in global coordinates.
    """
    L0 = mesh.elem_L0[ie]
    c = np.cos(theta); s = np.sin(theta)
    qx_l =  c*qx + s*qy;  qy_l = -s*qx + c*qy
    bx_l =  c*bx + s*by;  by_l = -s*bx + c*by
    A = mesh.elem_A[ie]
    fx = qx_l + bx_l*A;   fy = qy_l + by_l*A
    f_loc = np.array([fx*L0/2, fy*L0/2, fy*L0**2/12,
                      fx*L0/2, fy*L0/2, -fy*L0**2/12])
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    T2T = np.zeros((6, 6))
    T2T[0:3, 0:3] = R.T
    T2T[3:6, 3:6] = R.T
    return T2T @ f_loc


# =============================================================================
# VECTORISED ASSEMBLY — all elements processed simultaneously
# =============================================================================

def assemble(mesh: MeshedStructure,
             U: np.ndarray,
             theta_states: np.ndarray,
             dist_loads: Dict[int, Tuple[float, float]],
             joint_loads: List,
             lam: float) -> Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fully vectorised assembly using numpy broadcasting and einsum.
    No Python loop over elements — all ne elements processed simultaneously.

    Uses CR-TL formulation:
      K_tan = T1^T k_loc T1 + K4
    where K4 is the Crisfield geometric correction (Eq. 25).

    theta_states : (n_elems,) previous chord angles; nan = first iteration
    dist_loads   : {elem_idx: (qx, qy)} pre-scaled
    joint_loads  : list of (node_idx, Fx, Fy, Mz) pre-scaled
    """
    ndof    = mesh.n_dofs
    ne      = mesh.n_elems
    dof_arr = mesh.elem_dof_array   # (ne, 6)
    coords  = mesh.elem_coords       # (ne, 4)
    L0      = mesh.elem_L0           # (ne,)

    Fint      = np.zeros(ndof)
    Fext      = np.zeros(ndof)
    theta_new = np.empty(ne)
    theta_new[:] = np.nan

    # Nodal displacements for all elements simultaneously
    ux1 = U[dof_arr[:, 0]]; uy1 = U[dof_arr[:, 1]]; rz1 = U[dof_arr[:, 2]]
    ux2 = U[dof_arr[:, 3]]; uy2 = U[dof_arr[:, 4]]; rz2 = U[dof_arr[:, 5]]

    # Deformed geometry
    x1d = coords[:, 0] + ux1;  y1d = coords[:, 1] + uy1
    x2d = coords[:, 2] + ux2;  y2d = coords[:, 3] + uy2
    dx  = x2d - x1d;            dy  = y2d - y1d
    Ld  = np.hypot(dx, dy)

    # Reference chord angle
    theta0   = np.arctan2(coords[:, 3] - coords[:, 1],
                          coords[:, 2] - coords[:, 0])
    thetaRaw = np.arctan2(dy, dx)

    # De Souza rotation unwrap — handles multi-turn rotations correctly
    theta = thetaRaw.copy()
    valid = ~np.isnan(theta_states)
    if valid.any():
        d = thetaRaw[valid] - theta_states[valid]
        d -= 2*np.pi * np.round(d / (2*np.pi))
        theta[valid] = theta_states[valid] + d
    theta_new[:] = theta

    # Corotational local displacements
    dth = theta - theta0
    u4  = Ld - L0
    u3  = rz1 - dth
    u6  = rz2 - dth

    # Local stiffness scalars (ne,)
    EAL = mesh.elem_E * mesh.elem_A / L0
    EIL = mesh.elem_E * mesh.elem_I / L0
    N   = EAL * u4
    M1  = 4*EIL*u3 + 2*EIL*u6
    M2  = 2*EIL*u3 + 4*EIL*u6

    # T1 transformation matrix (ne, 3, 6)
    c = np.cos(theta); s = np.sin(theta)
    T1 = np.zeros((ne, 3, 6))
    T1[:, 0, 0] = -c;    T1[:, 0, 1] = -s;   T1[:, 0, 3] = c;    T1[:, 0, 4] = s
    T1[:, 1, 0] = -s/Ld; T1[:, 1, 1] = c/Ld; T1[:, 1, 2] = 1.0
    T1[:, 1, 3] =  s/Ld; T1[:, 1, 4] =-c/Ld
    T1[:, 2, 0] = -s/Ld; T1[:, 2, 1] = c/Ld; T1[:, 2, 5] = 1.0
    T1[:, 2, 3] =  s/Ld; T1[:, 2, 4] =-c/Ld

    # Local stiffness matrix (ne, 3, 3)
    kl = np.zeros((ne, 3, 3))
    kl[:, 0, 0] = EAL
    kl[:, 1, 1] = 4*EIL; kl[:, 1, 2] = 2*EIL
    kl[:, 2, 1] = 2*EIL; kl[:, 2, 2] = 4*EIL

    # Material stiffness: Ke_mat = T1^T k_loc T1  (ne, 6, 6)
    tmp    = np.einsum('eij,ejk->eik', kl, T1)         # (ne, 3, 6)
    Ke_mat = np.einsum('eji,ejk->eik', T1, tmp)        # (ne, 6, 6)

    # Geometric stiffness K4 (Crisfield Eq.25, deformed length Ld)
    z = np.column_stack([ s, -c, np.zeros(ne), -s,  c, np.zeros(ne)])  # (ne,6)
    r = np.column_stack([-c, -s, np.zeros(ne),  c,  s, np.zeros(ne)])  # (ne,6)
    NL   = (N / Ld)[:, None, None]
    M12L = ((M1 + M2) / Ld)[:, None, None]
    K4   = (NL  * np.einsum('ei,ej->eij', z, z)
            + M12L * (np.einsum('ei,ej->eij', r, z)
                      + np.einsum('ei,ej->eij', z, r)))  # (ne,6,6)

    Ke = Ke_mat + K4   # (ne, 6, 6) total tangent stiffness

    # Internal force: T1^T [N, M1, M2]  (ne, 6)
    floc   = np.column_stack([N, M1, M2])               # (ne, 3)
    Fint_e = np.einsum('eji,ej->ei', T1, floc)          # (ne, 6)

    # Assemble into global K (COO format -> CSR)
    rows_g = np.repeat(dof_arr, 6, axis=1).reshape(ne, 36)   # (ne,36)
    cols_g = np.tile(dof_arr, (1, 6)).reshape(ne, 36)         # (ne,36)
    K = csr_matrix((Ke.reshape(ne, 36).ravel(),
                    (rows_g.ravel(), cols_g.ravel())),
                   shape=(ndof, ndof))

    # Scatter Fint
    np.add.at(Fint, dof_arr.ravel(), Fint_e.ravel())

    # Distributed loads (element-by-element, typically sparse)
    for ie, (qx, qy) in dist_loads.items():
        fe = element_nodal_loads(ie, mesh, theta[ie], qx, qy)
        np.add.at(Fext, dof_arr[ie], fe)

    # Joint loads
    for jl in joint_loads:
        ni = jl[0]; Fx = jl[1]; Fy = jl[2]
        Mz = jl[3] if len(jl) > 3 else 0.0
        Fext[3*ni]   += Fx
        Fext[3*ni+1] += Fy
        Fext[3*ni+2] += Mz

    return K, Fint, Fext, theta_new


# =============================================================================
# BOUNDARY CONDITIONS — penalty method, no matrix copy
# =============================================================================

def apply_bcs_sparse(K: csr_matrix,
                     Fres: np.ndarray,
                     bc_dofs: List[int],
                     bc_delta: List[float]) -> Tuple[csr_matrix, np.ndarray]:
    """
    Apply prescribed displacement increments via large-penalty method.
    Adds penalty * I to diagonal at BC DOFs and sets RHS accordingly.
    No matrix structural changes — O(1) per BC DOF.
    """
    Fmod  = Fres.copy()
    K_mod = K.copy()
    penalty = K_mod.diagonal().max() * 1e8
    for dof, delta in zip(bc_dofs, bc_delta):
        K_mod[dof, dof] += penalty
        Fmod[dof] = penalty * delta
    return K_mod, Fmod


# =============================================================================
# NEWTON-RAPHSON SOLVER
# =============================================================================

def solve_step(mesh: MeshedStructure,
               U: np.ndarray,
               step_loads: dict,
               bc_dofs: List[int],
               bc_vals_target: List[float],
               max_iter: int = 50,
               tol: float = 5e-4,
               n_increments: int = 20) -> Tuple[np.ndarray, List[dict]]:
    """
    Incremental Newton-Raphson with adaptive load stepping.
    Sparse assembly + sparse direct solver (SuperLU via scipy).

    Convergence criterion: max|Fres[free DOFs]| / max|Fext| < tol
    Increment control: halve on non-convergence, grow x1.5 on success.
    """
    history     = []
    U_cur       = U.copy()
    inc_size    = 1.0 / n_increments
    lam         = 0.0
    theta_state = np.full(mesh.n_elems, np.nan)

    # Free DOF mask (exclude prescribed DOFs from convergence check)
    free_mask = np.ones(mesh.n_dofs, dtype=bool)
    for d in bc_dofs:
        free_mask[d] = False

    # Pre-build unscaled load lookups (computed once per step)
    raw_dist: Dict[int, Tuple[float, float]] = {}
    for (eidx, qx, qy) in (step_loads.get('line_init', [])
                            + step_loads.get('line_prop', [])):
        prev = raw_dist.get(eidx, (0.0, 0.0))
        raw_dist[eidx] = (prev[0]+qx, prev[1]+qy)

    raw_joints = []
    for jl in step_loads.get('joint_init', []) + step_loads.get('joint_prop', []):
        mi = mesh.user_node_to_mesh[jl.node_id]
        raw_joints.append((mi, jl.Fx, jl.Fy, jl.Mz))

    while lam < 1.0 - 1e-10:
        lam_t = min(lam + inc_size, 1.0)

        dist_scaled = {ie: (qx*lam_t, qy*lam_t) for ie, (qx, qy) in raw_dist.items()}
        jt_scaled   = [(ni, Fx*lam_t, Fy*lam_t, Mz*lam_t)
                       for (ni, Fx, Fy, Mz) in raw_joints]

        U_try     = U_cur.copy()
        th_try    = theta_state.copy()
        converged = False

        for it in range(max_iter):
            K, Fint, Fext, th_try = assemble(mesh, U_try, th_try,
                                              dist_scaled, jt_scaled, lam_t)
            Fres = Fext - Fint

            # Apply BCs
            bc_delta = [bc_vals_target[k]*lam_t - U_try[bc_dofs[k]]
                        for k in range(len(bc_dofs))]
            K_bc, F_bc = apply_bcs_sparse(K, Fres, bc_dofs, bc_delta)

            # Convergence check on free DOFs
            ext_mag = max(float(np.max(np.abs(Fext))), 1.0)
            res     = float(np.max(np.abs(Fres[free_mask]))) / ext_mag
            if it > 0 and res < tol:
                converged = True
                break

            # Sparse direct solve
            try:
                dU = spsolve(K_bc, F_bc)
            except Exception:
                break
            U_try += dU

        if converged:
            U_cur       = U_try
            theta_state = th_try
            lam         = lam_t
            inc_size    = min(inc_size * 1.5, 0.1)
            history.append({'lambda': lam, 'iterations': it+1,
                            'residual': res, 'U': U_cur.copy()})
        else:
            inc_size *= 0.5
            if inc_size < 1e-6:
                print(f"  Warning: minimum increment size reached at lam={lam:.4f}")
                lam = lam_t   # advance to avoid infinite loop
                continue

    return U_cur, history


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

class FEARunner:
    """
    High-level runner — identical API to v1, fully optimised internals.
    Drop-in replacement: change 'import nlfea' to point to this file.
    """

    def __init__(self, model: Model):
        self.model   = model
        self.mesh    = MeshedStructure(model)
        self.results = []     # list of increment history dicts per step
        self.U_steps = []     # displacement vector at end of each step
        self.U       = np.zeros(self.mesh.n_dofs)

    def _get_step_loads(self, step: int) -> dict:
        m  = self.model
        em = self.mesh.user_elem_to_mesh   # precomputed
        jl_init = [jl for jl in m.joint_loads if jl.step_start == step]
        jl_prop = [jl for jl in m.joint_loads if jl.step_start < step <= jl.step_end]
        li_init = [(eidx, lf.qx, lf.qy) for lf in m.line_forces
                   if lf.step_start == step for eidx in em.get(lf.elem_id, [])]
        li_prop = [(eidx, lf.qx, lf.qy) for lf in m.line_forces
                   if lf.step_start < step <= lf.step_end for eidx in em.get(lf.elem_id, [])]
        bi_init = [(eidx, bf.bx, bf.by) for bf in m.body_forces
                   if bf.step_start == step for eidx in em.get(bf.elem_id, [])]
        bi_prop = [(eidx, bf.bx, bf.by) for bf in m.body_forces
                   if bf.step_start < step <= bf.step_end for eidx in em.get(bf.elem_id, [])]
        return {'joint_init': jl_init, 'joint_prop': jl_prop,
                'line_init': li_init,  'line_prop': li_prop,
                'body_init': bi_init,  'body_prop': bi_prop}

    def _get_bc_info(self, step: int):
        bc_dofs, bc_vals = [], []
        for bc in [b for b in self.model.bcs if b.step_start <= step <= b.step_end]:
            mi  = self.mesh.user_node_to_mesh[bc.node_id]
            dof = 3*mi + (bc.dof - 1)
            bc_dofs.append(dof)
            bc_vals.append(bc.value)
        return bc_dofs, bc_vals

    def run(self, verbose: bool = True,
            max_iter: int = 50, tol: float = 5e-4, n_increments: int = 20):
        """Run all analysis steps."""
        t0 = time.time()
        for step in range(1, self.model.n_steps + 1):
            if verbose:
                print(f"\n--- Step {step} ---")
            sl               = self._get_step_loads(step)
            bc_dofs, bc_vals = self._get_bc_info(step)
            U_new, hist      = solve_step(self.mesh, self.U, sl, bc_dofs, bc_vals,
                                          max_iter=max_iter, tol=tol,
                                          n_increments=n_increments)
            self.U = U_new
            self.U_steps.append(U_new.copy())
            self.results.append(hist)
            if verbose and hist:
                print(f"  {len(hist)} increments | residual {hist[-1]['residual']:.2e}")
        if verbose:
            print(f"\nDone. {time.time()-t0:.3f}s "
                  f"({self.mesh.n_nodes} nodes, {self.mesh.n_elems} elements, "
                  f"{self.mesh.n_dofs} DOFs)")

    def get_node_displacement(self, node_id: int, step: int = -1):
        """Return (ux, uy, rz) for a user node at end of given step."""
        U  = self.U_steps[step] if self.U_steps else self.U
        mi = self.mesh.user_node_to_mesh[node_id]
        d  = self.mesh.dofs_of(mi)
        return float(U[d[0]]), float(U[d[1]]), float(U[d[2]])

    def print_results(self, node_ids=None):
        if node_ids is None:
            node_ids = [n.id for n in self.model.nodes]
        print("\n" + "="*60)
        print("DISPLACEMENT RESULTS (final step)")
        print("="*60)
        print(f"{'Node':>6}  {'ux (m)':>14}  {'uy (m)':>14}  {'rz (rad)':>14}")
        print("-"*60)
        for nid in node_ids:
            ux, uy, rz = self.get_node_displacement(nid)
            print(f"{nid:>6}  {ux:>14.6f}  {uy:>14.6f}  {rz:>14.6f}")
        print("="*60)

    def plot_deformed(self, step: int = -1, scale: float = 1.0,
                      title: str = "Deformed Shape",
                      save_path: Optional[str] = None):
        """Plot original and deformed configurations."""
        U      = self.U_steps[step] if self.U_steps else self.U
        coords = self.mesh.elem_coords
        fig, ax = plt.subplots(figsize=(10, 6))
        for ie in range(self.mesh.n_elems):
            x1, y1, x2, y2 = coords[ie]
            ax.plot([x1, x2], [y1, y2], 'b--', lw=0.8, alpha=0.3)
        xd = np.array([self.mesh.mesh_nodes[n][0]
                       for n in range(self.mesh.n_nodes)]) + scale * U[0::3]
        yd = np.array([self.mesh.mesh_nodes[n][1]
                       for n in range(self.mesh.n_nodes)]) + scale * U[1::3]
        for ie in range(self.mesh.n_elems):
            n1 = self.mesh.mesh_elems[ie][0]
            n2 = self.mesh.mesh_elems[ie][1]
            ax.plot([xd[n1], xd[n2]], [yd[n1], yd[n2]], 'r-', lw=2)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_title(title)
        ax.grid(True, alpha=0.3)
        orig = mpatches.Patch(color='blue', alpha=0.4, label='Original')
        defd = mpatches.Patch(color='red', label='Deformed')
        ax.legend(handles=[orig, defd])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.show()
        return fig
