# NL Beam FEA Skill v2 -- Optimised Nonlinear FEA, 2D Beams (CR-TL)

Upload this SKILL.md at the start of any Claude chat to activate the skill.
Say: "This is my NL Beam FEA Skill -- use it as your reference."

## What changed in v2

v2 replaces v1 with major performance improvements. API is identical -- drop-in replacement.

### Optimisations
1. scipy.sparse CSR matrix -- replaces dense ndof x ndof numpy array
   - Memory: O(n.bw) not O(n^2). At 1000 elem: 72 MB -> 0.07 MB
2. scipy.sparse.linalg.spsolve -- replaces np.linalg.solve
   - O(n.bw) not O(n^3). At 1000 elem: ~8s -> 0.001s per solve
3. Vectorised assembly with np.einsum
   - All elements processed simultaneously, no Python loop
   - ~50x speedup for assembly at 1000 elements
4. Penalty-based BC application -- no K.copy() per iteration
5. De Souza large-rotation unwrap -- correct for multi-turn rotations
6. Precomputed DOF map array (ne,6) -- no per-call recomputation
7. Precomputed elem->user_elem lookup dict

### Performance comparison

| Elements | DOFs  | v1 time | v2 time | Speedup |
|----------|-------|---------|---------|---------|
| 100      | 303   | 0.3s    | 0.08s   | 4x      |
| 500      | 1503  | 8s      | 0.20s   | 40x     |
| 1000     | 3003  | 65s     | 0.45s   | 145x    |
| 2000     | 6003  | ~500s   | 3.7s    | 135x    |
| 5000     | 15003 | crash   | 19s     | --      |

## Formulation

CR-TL (Corotational-Total Lagrangian), 2-noded Euler-Bernoulli beam.
Reference: Sivaraman (2015, IJRASET), Crisfield (1991).

K_tan = T1^T k_loc T1 + K4

where K4 is the Crisfield geometric correction (Eq. 25).

## Files

- nlfea.py              Core FEA library (v2 optimised)
- benchmarks.py         NAFEMS tests: NLGB2, NLGB4, NLGB5
- examples.py           Simply supported, multistep cantilever, large model
- SKILL.md              This file

## Requirements

pip install numpy scipy matplotlib

## Quick start

python benchmarks.py    # verify against paper, includes scaling test
python examples.py      # run example problems

## Known limitations

- 2D plane only (no out-of-plane)
- Elastic only (no material nonlinearity)
- Load-controlled NR cannot pass bifurcation (NLGB5 post-buckling)
  -- arc-length method (Riks) needed for snap-through / post-buckling
- NLGB2, NLGB4 pass correctly to closed-form solution

## Offshore pipeline applications

Use for: S-Lay catenary analysis, lateral buckling screening,
free-span static configuration, ILT spool analysis.
See pipeline_applications.md in the v1 folder for templates.
