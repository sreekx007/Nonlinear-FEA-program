"""
Benchmarks — NL Beam FEA Optimised (v2)
========================================
NAFEMS NLGB2, NLGB4, NLGB5 + scaling benchmark.
Run:  python benchmarks.py

Expected results:
  NLGB2 (16 elem): UA/L=-1.0000, VA/L=0.0000, tA/2pi=1.0000  Time ~0.04s
  NLGB4 ( 8 elem): UA/L=-0.5549, VA/L=-0.8131, tA/(pi/2)=-0.9122
  NLGB5 (16 elem): NOTE - requires arc-length for post-buckling.
                   Load-controlled NR gives pre-buckling solution only.

Scaling (NLGB2 full circle):
   100 elem  303 DOFs  ~0.08s
   500 elem 1503 DOFs  ~0.20s
  1000 elem 3003 DOFs  ~0.45s
  2000 elem 6003 DOFs  ~3.7s
"""

import numpy as np
import time
from nlfea import (Model, Node, Section, Material, UserElement,
                   JointLoad, BoundaryCondition, FEARunner)

L = 3.2; b = d = 0.1; E = 210e9
I = b*d**3/12; EI = E*I


def build_cantilever(n_elem):
    nodes    = [Node(i+1, i*L/n_elem, 0.0) for i in range(n_elem+1)]
    sections = [Section(1, b, d)]
    mats     = [Material(1, E)]
    elements = [UserElement(i+1, i+1, i+2, 1, 1) for i in range(n_elem)]
    bcs      = [BoundaryCondition(i+1, 1, i+1, 0.0, 1, 1) for i in range(3)]
    return Model(nodes=nodes, elements=elements, sections=sections,
                 materials=mats, bcs=bcs, n_steps=1)


def test_NLGB2(n_elem=16):
    print(f"\n{'='*60}")
    print(f"NLGB2 -- End Moment Cantilever  ({n_elem} elements)")
    print(f"{'='*60}")
    M_full = 2*np.pi*EI/L
    model  = build_cantilever(n_elem)
    model.joint_loads = [JointLoad(1, n_elem+1, 0, 0, M_full, 1, 1)]
    r = FEARunner(model)
    t0 = time.time(); r.run(verbose=False, n_increments=50); elapsed = time.time()-t0
    ux, uy, rz = r.get_node_displacement(n_elem+1)
    print(f"  UA/L   = {ux/L:+.4f}   (closed form: -1.000)")
    print(f"  VA/L   = {uy/L:+.4f}   (closed form:  0.000)")
    print(f"  tA/2pi = {rz/(2*np.pi):+.4f}   (closed form:  1.000)")
    print(f"  Time   = {elapsed:.3f} s")
    return r


def test_NLGB4(n_elem=8):
    print(f"\n{'='*60}")
    print(f"NLGB4 -- Transverse Load  ({n_elem} elements)")
    print(f"{'='*60}")
    P = 10*EI/L**2
    model = build_cantilever(n_elem)
    model.joint_loads = [JointLoad(1, n_elem+1, 0, -P, 0, 1, 1)]
    r = FEARunner(model)
    t0 = time.time(); r.run(verbose=False); elapsed = time.time()-t0
    ux, uy, rz = r.get_node_displacement(n_elem+1)
    print(f"  UA/L      = {ux/L:+.4f}   (closed form: -0.555)")
    print(f"  VA/L      = {uy/L:+.4f}   (closed form: -0.811)")
    print(f"  tA/(pi/2) = {rz/(np.pi/2):+.4f}   (closed form: -0.911)")
    print(f"  Time      = {elapsed:.3f} s")
    return r


def test_NLGB5(n_elem=16):
    print(f"\n{'='*60}")
    print(f"NLGB5 -- Axial Load / Bifurcation  ({n_elem} elements)")
    print(f"{'='*60}")
    P = 22.493*EI/L**2; Q = P/1000
    model = build_cantilever(n_elem)
    model.joint_loads = [JointLoad(1, n_elem+1, -P, -Q, 0, 1, 1)]
    r = FEARunner(model)
    t0 = time.time(); r.run(verbose=False); elapsed = time.time()-t0
    ux, uy, rz = r.get_node_displacement(n_elem+1)
    print(f"  UA/L      = {ux/L:+.4f}   (closed form: -1.577)")
    print(f"  VA/L      = {uy/L:+.4f}   (closed form: -0.421)")
    print(f"  tA/(pi/2) = {rz/(np.pi/2):+.4f}   (closed form: -0.978)")
    print(f"  Time      = {elapsed:.3f} s")
    print(f"  NOTE: NLGB5 is a post-buckling problem. Load-controlled NR")
    print(f"        gives pre-buckling solution. Arc-length needed for full result.")
    return r


def scaling_benchmark():
    print(f"\n{'='*60}")
    print("SCALING BENCHMARK -- v2 (sparse solver)")
    print(f"{'='*60}")
    print(f"{'Elements':>10}  {'DOFs':>6}  {'Time (s)':>10}  {'UA/L':>8}  {'err':>8}")
    print("-"*50)
    M_full = 2*np.pi*EI/L
    for n in [50, 100, 200, 500, 1000, 2000]:
        model = build_cantilever(n)
        model.joint_loads = [JointLoad(1, n+1, 0, 0, M_full, 1, 1)]
        r = FEARunner(model)
        t0 = time.time()
        r.run(verbose=False, n_increments=50)
        elapsed = time.time()-t0
        ux, uy, rz = r.get_node_displacement(n+1)
        err = abs(ux/L + 1.0)
        print(f"{n:>10}  {3*(n+1):>6}  {elapsed:>10.3f}  {ux/L:>8.4f}  {err:>8.4f}")


if __name__ == '__main__':
    print("NL Beam FEA -- v2 Optimised -- NAFEMS Benchmarks")
    test_NLGB2(n_elem=16)
    test_NLGB4(n_elem=8)
    test_NLGB5(n_elem=16)
    scaling_benchmark()
    print("\nAll benchmarks complete.")
