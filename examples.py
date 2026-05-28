"""
Examples -- NL Beam FEA Optimised (v2)
Run:  python examples.py
"""

import numpy as np, time
from nlfea import (Model, Node, Section, Material, UserElement,
                   JointLoad, LineForce, BoundaryCondition, FEARunner)


def example_simply_supported():
    print("\n" + "="*55)
    print("Example 1: Simply Supported Beam -- Uniform Load")
    print("="*55)
    L=5.0; b=0.2; d=0.3; E=200e9; w=-50000.0; n=8
    nodes    = [Node(i+1, i*L/n, 0.0) for i in range(n+1)]
    sections = [Section(1, b, d)]; mats = [Material(1, E)]
    elements = [UserElement(i+1, i+1, i+2, 1, 1) for i in range(n)]
    bcs      = [BoundaryCondition(1,1,1,0.0,1,1),
                BoundaryCondition(2,1,2,0.0,1,1),
                BoundaryCondition(3,n+1,2,0.0,1,1)]
    lf       = [LineForce(i+1, i+1, 0.0, w, 1, 1) for i in range(n)]
    model    = Model(nodes=nodes, elements=elements, sections=sections,
                     materials=mats, bcs=bcs, line_forces=lf, n_steps=1)
    runner   = FEARunner(model); runner.run(); runner.print_results()
    ux, uy, rz = runner.get_node_displacement(n//2+1)
    EI = E*b*d**3/12; uy_lin = -5*abs(w)*L**4/(384*EI)
    print(f"\nMidspan: {uy:.5f} m | Linear: {uy_lin:.5f} m | NL/L: {uy/uy_lin:.4f}")
    return runner


def example_cantilever_multistep():
    print("\n" + "="*55)
    print("Example 2: Cantilever -- Two Load Steps")
    print("="*55)
    L=3.2; b=0.1; d=0.1; E=210e9; EI=E*b*d**3/12; n=8
    nodes    = [Node(i+1, i*L/n, 0.0) for i in range(n+1)]
    sections = [Section(1, b, d)]; mats = [Material(1, E)]
    elements = [UserElement(i+1, i+1, i+2, 1, 1) for i in range(n)]
    bcs      = [BoundaryCondition(i+1, 1, i+1, 0.0, 1, 2) for i in range(3)]
    P1 = 5*EI/L**2; M2 = np.pi*EI/L
    jl = [JointLoad(1, n+1, 0.0, P1,  0.0, step_start=1, step_end=2),
          JointLoad(2, n+1, 0.0, 0.0, M2,  step_start=2, step_end=2)]
    model  = Model(nodes=nodes, elements=elements, sections=sections,
                   materials=mats, bcs=bcs, joint_loads=jl, n_steps=2)
    runner = FEARunner(model); runner.run(); runner.print_results()
    return runner


def example_large_model():
    print("\n" + "="*55)
    print("Example 3: Large Model -- 500 elements (v2 speed test)")
    print("="*55)
    L=3.2; b=0.1; d=0.1; E=210e9; EI=E*b*d**3/12; n=500
    nodes    = [Node(i+1, i*L/n, 0.0) for i in range(n+1)]
    sections = [Section(1, b, d)]; mats = [Material(1, E)]
    elements = [UserElement(i+1, i+1, i+2, 1, 1) for i in range(n)]
    bcs      = [BoundaryCondition(i+1, 1, i+1, 0.0, 1, 1) for i in range(3)]
    model    = Model(nodes=nodes, elements=elements, sections=sections,
                     materials=mats, bcs=bcs,
                     joint_loads=[JointLoad(1, n+1, 0, 0, 2*np.pi*EI/L, 1, 1)],
                     n_steps=1)
    runner = FEARunner(model)
    t0 = time.time(); runner.run(n_increments=50); elapsed = time.time()-t0
    ux, uy, rz = runner.get_node_displacement(n+1)
    print(f"\n500-element full circle (1503 DOFs):")
    print(f"  UA/L  = {ux/L:+.4f}  (expected -1.000)")
    print(f"  VA/L  = {uy/L:+.4f}  (expected  0.000)")
    print(f"  tA/2pi= {rz/(2*np.pi):+.4f}  (expected  1.000)")
    print(f"  Time  = {elapsed:.2f} s")
    return runner


if __name__ == '__main__':
    example_simply_supported()
    example_cantilever_multistep()
    example_large_model()
    print("\nAll examples complete.")
