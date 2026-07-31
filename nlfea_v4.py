"""
NL Beam FEA - Version 4 (Incremental Elastoplasticity)
=======================================================
Nonlinear FEA for large-displacement 2D beam structures.
Based on: Sivaraman (2015, IJRASET) - CR-TL formulation.

v4.1 fix (Session 19, 15 Jul 2026):
  19. get_element_strains() BUG FIX: extreme-fibre radius r_o was
      reconstructed from fibre spacing (max|fy| + 0.5*(fy[1]-fy[0])),
      exact ONLY for the Cartesian fibre model (PipeSection, evenly-
      spaced strips span exactly to r_o by construction). For
      PipeSectionPolar (angular points at r_mid*sin(theta), unevenly
      spaced, never reaching r_o), this silently returned ~1.28x the
      true r_o -- verified numerically at 260.8mm reconstructed vs.
      203.2mm true r_o for a 16in x 21mm pipe -- inflating every
      reported strain and stress for polar-section elements by ~28%,
      in both the R-O (elem_inelastic) and EP (elem_ep) branches.
      Fixed: both branches now use sec.r_o directly (a property already
      exposed by PipeSection and inherited unchanged by PipeSectionPolar).
      This is section-type-agnostic and exactly reproduces the previous
      (correct) value for the Cartesian fibre case -- no change to any
      previously validated fibre-model benchmark result. Found while
      wiring PipeSectionPolar into an external driver (slay_overbend);
      not caught by this paper's own benchmarks because none of them
      exercised get_element_strains() with a PipeSectionPolar section.

v4 adds over v3:
  13. IncrementalIsotropic material -- J2 plasticity, isotropic hardening.
      Linear or piecewise-linear (tabulated) hardening.
      from_ro() classmethod converts R-O curve to tabulated hardening.
  14. PlasticState -- per-element / per-Gauss-point / per-fibre plastic
      strain state.  Persists across Newton iterations and load increments.
      Optionally carried across run_slay() calls (passage simulation).
  15. _ep_return_mapping() -- 1D radial return mapping, vectorised over fibres.
      Uses COMMITTED plastic state (frozen at increment start); Newton-safe.
  16. assemble() extended: optional plastic_state argument; always returns
      5-tuple (K, Fint, Fext, theta_new, plastic_state_new).
  17. solve_step() extended: commits plastic state each increment; returns
      (U, history, plastic_state_final).
  18. FEARunner.run() initialises and propagates plastic state.

v3 additions over v2:
  8. Ramberg-Osgood material model (DNV form) - smooth nonlinear sigma-eps curve
  9. Fibre/layer cross-section integration - distributed plasticity through depth
  10. Circular Hollow Section (PipeSection) - D_o, t, n_fibres
  11. Tangent section moduli EA_t, ES_t, EI_t updated each NR iteration
  12. get_element_strains() - extreme fibre strains for strain-based design

Material model - Ramberg-Osgood, two supported forms:

  DNV form (general):
    eps / eps_y = sig / sig_y  +  alpha * (sig / sig_y)^n
    where eps_y = sig_y / E

  API FFS 579-1 / CSA Z662 pipeline form:
    eps = sig/E + (0.005 - sig_ys/E) * (sig/sig_ys)^N_RO    [Eq 2E.48]
    N_RO derived from sig_ys/sig_uts ratio via Eq 2E.49/2E.50/2E.52.
    Converted to DNV alpha by: alpha_DNV = (0.005 - sig_ys/E) * E / sig_ys
    Internally stored as DNV form in all cases.

Tangent modulus:
  E_t(sig) = E / (1 + alpha * n * (sig / sig_y)^(n-1))

Project reference - DNV Grade 450, 16" x 21mm seamless:
  Source: API FFS 579-1/ASME FFS-1 2021, Annex 2E, Eq 2E.48 (Session 15)
  SMYS = 450 MPa, E = 207 GPa, N_RO = 20.59
  Validated against Abaqus tabular data (Steel_NL, E=210 GPa).
  Use: RambergOsgood.dnv450()

Backward compatibility:
  Material(id, E)  +  Section(id, b, d)  -> elastic rectangular (unchanged)
  RambergOsgood    +  PipeSection        -> inelastic CHS (new)

v2 performance retained (sparse + vectorised assembly).

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
# DATA STRUCTURES  (unchanged from v1 - fully compatible)
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
class PipeSection:
    """
    Circular Hollow Section for fibre integration.

    Fibres are horizontal strips through the annulus, equally spaced
    over the outer diameter. Used with RambergOsgood material.

    Parameters
    ----------
    id       : section id
    D_o      : outer diameter (m)
    t        : wall thickness (m)
    n_fibres : number of integration fibres (default 20; 30+ for high n)
    """
    id:       int
    D_o:      float
    t:        float
    n_fibres: int = 20

    @property
    def D_i(self): return self.D_o - 2.0 * self.t

    @property
    def r_o(self): return self.D_o / 2.0

    @property
    def r_i(self): return self.D_i / 2.0

    @property
    def A(self):
        return np.pi / 4.0 * (self.D_o**2 - self.D_i**2)

    @property
    def I(self):
        return np.pi / 64.0 * (self.D_o**4 - self.D_i**4)

    def fibre_geometry(self):
        """
        Return (y_centres, areas) for n_fibres equal-width horizontal strips
        spanning -r_o to +r_o.

        Each strip i:
            y_i  = centroid of strip i
            A_i  = area of annular slice at y_i (outer chord - inner chord) * dy

        Returns
        -------
        y : (n_fibres,)  fibre centroid y-coordinates
        A : (n_fibres,)  fibre areas
        """
        ro = self.r_o
        ri = self.r_i
        nf = self.n_fibres
        dy = 2.0 * ro / nf
        y  = np.linspace(-ro + 0.5*dy, ro - 0.5*dy, nf)

        def chord(r, yi):
            arg = r**2 - yi**2
            return 2.0 * np.sqrt(np.maximum(arg, 0.0))

        b_fib = chord(ro, y) - chord(ri, y)
        A_fib = b_fib * dy
        return y, A_fib

@dataclass
class PipeSectionPolar(PipeSection):
    """
    B31-equivalent CHS: thin-wall POLAR (angular) integration at r_mid.

    Replicates the Abaqus B31 PIPE section integration scheme:
      - all wall material collapsed onto a single ring at
            r_mid = (r_o + r_i) / 2
      - n_points integration points equally spaced around the
        circumference, each with area A_wall / n_points
      - ALIGNED placement by default (offset_deg = 0): point 1 at
        theta = 0 on the bending neutral axis, exactly as Abaqus
        places PIPE section points. Two points (theta = 0, 180 deg)
        then contribute nothing to bending.

    Properties of the aligned 8-point scheme (verified vs Abaqus
    BM_Test_B ROTCTRL SM/SK data to +/-0.00% at all curvatures):
      - Elastic I integrates EXACTLY  (sum sin^2 exact at N>=3)
        -> I = A_wall * r_mid^2 / 2 = pi * r_mid^3 * t  (thin wall,
           0.30% below the exact thick-CHS value)
      - Fully-plastic modulus 5.2% LOW (trapezoid on |sin| with a
        point on the kink) -> over-predicts plastic curvature by up
        to ~45% near the plastic plateau, exactly as Abaqus B31 does.

    Use for Abaqus-compatibility benchmarking. For accurate section
    response use the parent PipeSection (Cartesian strips, converged).

    Parameters
    ----------
    id         : section id
    D_o        : outer diameter (m)
    t          : wall thickness (m)
    n_fibres   : number of angular integration points (default 8 = B31)
    offset_deg : angular offset of point 1 from the neutral axis in
                 degrees (default 0.0 = Abaqus-aligned; 22.5 gives the
                 staggered midpoint rule, which OVERSHOOTS Z_p by 2.6%)
    """
    offset_deg: float = 0.0

    @property
    def r_mid(self):
        return 0.5 * (self.r_o + self.r_i)

    @property
    def I(self):
        # Thin-wall discrete-consistent value: A_wall * r_mid^2 / 2.
        # Matches what the angular fibres themselves integrate, and the
        # value Abaqus B31 PIPE uses elastically (pi * r_mid^3 * t).
        return 0.5 * self.A * self.r_mid**2

    def fibre_geometry(self):
        """
        Return (y_centres, areas) for n_fibres angular points on the
        r_mid circle:  y_k = r_mid * sin(theta_k),  A_k = A_wall / N,
        theta_k = offset + k * 2*pi/N.
        """
        N  = self.n_fibres
        th = np.radians(self.offset_deg) + np.arange(N) * 2.0 * np.pi / N
        y  = self.r_mid * np.sin(th)
        A  = np.full(N, self.A / N)
        return y, A

@dataclass
class Material:
    id: int
    E: float    # Young's modulus (Pa)
    nu: float = 0.0

@dataclass
class RambergOsgood:
    """
    Ramberg-Osgood material model.

    Stored internally in DNV form:
        eps / eps_y  =  sig / sig_y  +  alpha * (sig / sig_y)^n
        eps_y = sig_y / E

    Constructor for pipeline steel (API FFS 579-1 form):
        eps = sig/E + (0.005 - sig_ys/E) * (sig/sig_ys)^N_RO   [Eq 2E.48]
        Use: RambergOsgood.from_ffs579(id, E, sig_ys)
        alpha_DNV = (0.005 - sig_ys/E) * E / sig_ys

    Tangent modulus:
        E_t(sig) = E / ( 1 + alpha * n * (|sig| / sig_y)^(n-1) )

    Project reference - DNV Grade 450, 16" x 21mm seamless:
        RambergOsgood.dnv450()
    """
    id:    int
    E:     float   # Young's modulus (Pa)
    sig_y: float   # Yield / reference stress (Pa)
    alpha: float   # Strain hardening coefficient - DNV convention
    n:     float   # Strain hardening exponent

    #        alternative constructors

    @classmethod
    def from_ffs579(cls, id: int, E: float, sig_ys: float):
        """
        Construct from API FFS 579-1/ASME FFS-1 2021 Annex 2E pipeline model.

        Engineering stress-strain curve for API 5L Grades X52-X80 (§2E.3.5.2):
            eps = sig/E + (0.005 - sig_ys/E) * (sig/sig_ys)^N_RO    [Eq 2E.48]

        N_RO derived from yield-to-tensile ratio (Eq 2E.49/2E.50/2E.52):
            sig_ys/sig_uts = 1 / (1 + 2*(149.96/sig_ys_MPa)^2.3)    [Eq 2E.50]
            eps_uts = -0.000254*sig_ys_MPa + 0.22                    [Eq 2E.52]
            N_RO = ln(eps_uts/0.005) / ln(sig_uts/sig_ys)            [Eq 2E.49]

        Converted to DNV internal form:
            alpha_DNV = (0.005 - sig_ys/E) * E / sig_ys
            n_DNV     = N_RO
            sig_y_DNV = sig_ys

        Validation (Session 15, 5 Jun 2026):
            sig_ys=450 MPa, E=207 GPa --> N_RO=20.59, alpha_DNV=2.278
            0.2% proof stress = 442.5 MPa  (SMYS target: 450 MPa)
            Matches Abaqus tabular (Steel_NL, E=210 GPa) within ~5-8 MPa
            above 0.5% strain; agrees within ~3% at 1.0% strain.

        Parameters
        ----------
        id     : material id
        E      : Young's modulus (Pa)
        sig_ys : SMYS or representative yield strength (Pa)
        """
        import numpy as np
        sig_ys_mpa = sig_ys / 1e6
        # Eq 2E.50 -- yield/tensile ratio
        ratio = 1.0 / (1.0 + 2.0 * (149.96 / sig_ys_mpa) ** 2.3)
        sig_uts = sig_ys / ratio
        # Eq 2E.52 -- total strain at UTS
        eps_uts = -0.000254 * sig_ys_mpa + 0.22
        # Eq 2E.49 -- N_RO
        N_RO = np.log(eps_uts / 0.005) / np.log(1.0 / ratio)
        # Convert to DNV alpha
        alpha_dnv = (0.005 - sig_ys / E) * E / sig_ys
        return cls(id=id, E=E, sig_y=sig_ys, alpha=alpha_dnv, n=N_RO)

    @classmethod
    def dnv450(cls, id: int = 1):
        """
        DNV Grade 450 pipeline steel -- project reference parameters.

        Source: API FFS 579-1/ASME FFS-1 2021, Annex 2E, Eq 2E.48 (Session 15)
                Engineering stress-strain model for API 5L Grades X52-X80.
                SMYS = 450 MPa, E = 207 GPa.

        Derived parameters (Eq 2E.49/2E.50/2E.52):
            sig_ys/sig_uts = 0.8623  -->  sig_uts = 521.9 MPa
            eps_uts = 10.57%
            N_RO = 20.59
            alpha_DNV = (0.005 - 450e6/207e9) * 207e9/450e6 = 1.300

        Stress-strain curve properties:
            0.2% proof stress = 442.5 MPa
            At 0.5% total strain: sigma = 450.0 MPa (= SMYS, by construction)
            At 1.0% total strain: sigma = 472.5 MPa

        Validation against Abaqus (Steel_NL, E=210 GPa, tabular):
            Agrees within ~5-8 MPa above 0.5% strain.

        Pipe geometry this was derived for:
            OD = 406.4 mm (16"), WT = 21 mm, seamless, DNV Grade 450 Medium SC

        DNV-ST-F101 capacity (Medium SC, empty overbend, zero tension):
            DCC: eps_Rd = 1.543%,  M = 1453 kN.m
            LCC: M_char = 1380 kN.m  (governs at all SC)
        """
        return cls.from_ffs579(id=id, E=207e9, sig_ys=450e6)

    @classmethod
    def abaqus_steel_nl(cls, id: int = 1):
        """
        DNV 450 steel as defined in Abaqus Steel_NL material (benchmark reference).

        Source: Stress-Strain Curve.Rev A.xlsx, tab 'DNV 450' (installation spec).
                Parameters from material header (image 2, rows 12-19):
                    SMYS  = 448 MPa  (actual certificate value)
                    SMTS  = 535 MPa  (actual certificate value)
                    E     = 210 GPa  (Abaqus Steel_NL definition)
                    n     = 17.22    (from actual Y/T = 448/535 = 0.8374 via Eq A-11a)

        Equation (A-10 / FFS 579-1 Eq 2E.48, same form):
            eps = sig/E + (0.005 - sig_y/E) * (sig/sig_y)^n

        Derived parameters:
            Y/T actual   = 448/535 = 0.8374  (material certificate)
            Y/T formula  = 0.8610            (Eq A-11b, estimated)
            alpha_DNV    = (0.005 - 448e6/210e9) * 210e9/448e6 = 1.344
            n            = 17.22  (used directly from certificate Y/T)

        Stress-strain curve properties:
            0.2% proof stress ≈ 385 MPa  (read from tabular data at eps_p=0.002)
            At 0.5% total strain: sigma ≈ 440 MPa
            At 1.0% total strain: sigma ≈ 460 MPa
            At 5.44% total strain: sigma = 530 MPa (last tabular point)

        Difference from dnv450():
            dnv450()         uses E=207 GPa, SMYS=450 MPa, Y/T estimated -> n=20.59
            abaqus_steel_nl() uses E=210 GPa, SMYS=448 MPa, Y/T actual   -> n=17.22
            Lower n = more gradual transition, lower 0.2% proof stress (385 vs 442 MPa)

        Use: benchmark reference only. All project parametric analyses use dnv450().
        """
        E      = 210e9
        sig_ys = 448e6
        n      = 17.22
        alpha  = (0.005 - sig_ys / E) * E / sig_ys   # = 1.344
        return cls(id=id, E=E, sig_y=sig_ys, alpha=alpha, n=n)


@dataclass
class IncrementalIsotropic:
    """
    Incremental J2 plasticity with isotropic hardening (v4).

    1D radial return mapping per fibre:
        sigma_trial = E * (eps_total - eps_p_committed)
        if |sigma_trial| > sigma_y(kap):
            d_gamma  = (|sigma_trial| - sigma_y(kap)) / (E + H_cur)
            eps_p   += d_gamma * sign(sigma_trial)
            kap     += d_gamma
            sigma    = sigma_y(kap) * sign(sigma_trial)
            E_t      = E * H_cur / (E + H_cur)   [consistent tangent]
        else:
            sigma = sigma_trial,  E_t = E

    Hardening options
    -----------------
    Linear:     sigma_y(kap) = sigma_y0 + H * kap
    Tabulated:  piecewise-linear (eps_p_table, sigma_y_table)
                -- use from_ro() to build from a RambergOsgood curve.

    Key difference from RambergOsgood
    ----------------------------------
    R-O: deformation plasticity -- sigma = f(eps_total), path-independent.
    EP:  incremental plasticity -- sigma depends on eps_p history, path-dependent.
    Monotonic first-loading: both give the same response when the hardening
    table is built from the R-O curve (see from_ro()).
    """
    id:      int
    E:       float   # Young's modulus (Pa)
    sigma_y0: float  # Initial yield stress (Pa) -- usually SMYS
    H:       float   # Plastic tangent modulus for linear branch (Pa)
                     # used when eps_p_table is None, or as fallback beyond table
    eps_p_table:    np.ndarray = None   # plastic strain breakpoints
    sigma_y_table:  np.ndarray = None   # yield stress at each breakpoint (Pa)

    # -- convenience constructors ------------------------------------------

    @classmethod
    def from_ro(cls, id: int, ro: 'RambergOsgood', n_pts: int = 40):
        """
        Build tabulated IncrementalIsotropic from a RambergOsgood material.

        The initial yield is set to the 0.2% proof stress (sigma where R-O
        plastic strain = 0.2%).  The hardening table is the R-O eps_p vs sigma
        curve shifted so eps_p = 0 at initial yield.  This makes the
        incremental model match R-O on FIRST MONOTONIC LOADING from zero.

        Parameters
        ----------
        ro    : RambergOsgood material to convert
        n_pts : number of table points above initial yield (default 40)

        Derivation
        ----------
        At sigma_y0 (0.2% proof stress):
            eps_p0 = alpha*(sigma_y0/sig_y)^n * sig_y/E  = 0.002 by definition

        Hardening table: for sigma in [sigma_y0, sigma_max]:
            eps_p_table[i] = R-O eps_p(sigma[i]) - eps_p0   (shifted to zero at yield)
            sig_y_table[i] = sigma[i]

        R-O tangent plastic modulus at initial yield (used as H fallback):
            E_t0  = E / (1 + alpha * n * (sigma_y0/sig_y)^(n-1))
            H_lin = E * E_t0 / (E - E_t0)
        """
        # 0.2% proof stress: R-O eps_p = 0.002  →  solve for sigma_y0
        #   alpha*(sig/sig_y)^n * sig_y/E = 0.002
        #   sig/sig_y = (0.002 * E / (alpha * sig_y))^(1/n)
        val = 0.002 * ro.E / (ro.alpha * ro.sig_y)
        sig_y0 = ro.sig_y * (val ** (1.0 / ro.n)) if val < 1.0 else ro.sig_y
        eps_p0 = ro.alpha * (sig_y0 / ro.sig_y)**ro.n * ro.sig_y / ro.E

        # Sample sigma from sig_y0 to sigma_max
        sigma_max = 1.5 * ro.sig_y
        sigma_arr = np.linspace(sig_y0, sigma_max, n_pts)
        eps_p_arr = (ro.alpha * (sigma_arr / ro.sig_y)**ro.n * ro.sig_y / ro.E
                     - eps_p0)                  # shifted: 0 at sig_y0
        eps_p_arr = np.maximum(eps_p_arr, 0.0)  # guard against floating-point negatives

        # Consistent tangent plastic modulus at initial yield
        abs_sn = (sig_y0 / ro.sig_y)**(ro.n - 1.0)
        E_t0   = ro.E / (1.0 + ro.alpha * ro.n * abs_sn)
        H_lin  = ro.E * E_t0 / max(ro.E - E_t0, 1.0)

        return cls(id=id, E=ro.E, sigma_y0=sig_y0, H=H_lin,
                   eps_p_table=eps_p_arr, sigma_y_table=sigma_arr)

    @classmethod
    def dnv450(cls, id: int = 1):
        """DNV Grade 450 incremental plasticity from FFS 579-1 R-O curve."""
        ro = RambergOsgood.from_ffs579(id=id, E=207e9, sig_ys=450e6)
        return cls.from_ro(id=id, ro=ro)

    @classmethod
    def abaqus_steel_nl(cls, id: int = 1):
        """
        DNV 450 incremental plasticity built from Abaqus Steel_NL R-O curve.

        Converts RambergOsgood.abaqus_steel_nl() (E=210 GPa, SMYS=448 MPa,
        n=17.22) into an incremental hardening table via from_ro().

        On monotonic first loading the curve is identical to the R-O version.
        On unloading and reloading the incremental model correctly returns
        elastically -- which is what the Abaqus load-unload benchmark tests.

        Use this material for all load-unload benchmark comparisons.
        Use dnv450() for all project parametric analyses (monotonic only).
        """
        ro = RambergOsgood.abaqus_steel_nl(id=id)
        return cls.from_ro(id=id, ro=ro)

    @classmethod
    def abaqus_steel_nl_tabular(cls, id: int = 1):
        """
        DNV 450 incremental plasticity from the direct true stress / true
        strain spreadsheet data (Stress-Strain Curve.Rev A.xlsx, DNV 450 tab).

        Use this material when comparing against Abaqus results that were
        run with the *PLASTIC table supplied from the same spreadsheet.

        Source data: 30 rows (true total strain, true stress in MPa).
        E        = 210 GPa
        sigma_y0 = 385 MPa  -- initial yield (first non-linear data point)

        True plastic strain = true_total_strain - true_stress / E,
        shifted so the first entry has eps_p = 0.

        Contrast with abaqus_steel_nl():
            abaqus_steel_nl()         sigma_y0 = 438.7 MPa (0.2% proof, R-O)
            abaqus_steel_nl_tabular() sigma_y0 = 385.0 MPa (spreadsheet row 1)
        The lower initial yield gives earlier onset of plasticity and more
        rotation for a given applied moment.
        """
        E_Pa  = 210e9
        E_MPa = 210000.0

        # True stress-strain pairs from spreadsheet rows 43-72
        data = np.array([
            [0.0020, 385], [0.0021, 390], [0.0022, 395], [0.0023, 400],
            [0.0024, 405], [0.0026, 410], [0.0027, 415], [0.0029, 420],
            [0.0032, 425], [0.0035, 430], [0.0038, 435], [0.0042, 440],
            [0.0047, 445], [0.0052, 450], [0.0059, 455], [0.0067, 460],
            [0.0077, 465], [0.0088, 470], [0.0101, 475], [0.0117, 480],
            [0.0136, 485], [0.0158, 490], [0.0183, 495], [0.0214, 500],
            [0.0250, 505], [0.0292, 510], [0.0341, 515], [0.0398, 520],
            [0.0465, 525], [0.0544, 530],
        ])
        sigma     = data[:, 1] * 1e6                    # Pa
        eps_p_raw = data[:, 0] - data[:, 1] / E_MPa    # true plastic strain
        eps_p_tbl = eps_p_raw - eps_p_raw[0]            # shift: row 1 → 0

        return cls(id=id, E=E_Pa, sigma_y0=float(sigma[0]), H=0.0,
                   eps_p_table=eps_p_tbl, sigma_y_table=sigma)

    @classmethod
    def abaqus_steel_nl_verified(cls, id: int = 1):
        """
        VERIFIED material -- transcribed directly from the *PLASTIC block of
        the author's actual original Abaqus benchmark model (BM_Test_B, the
        one that produced the reference theta = 0.266/0.395/0.624/1.031 rad
        at M = 1300/1350/1400/1450 kNm). Confirmed by direct screenshot of
        the Abaqus/CAE keyword editor, July 2026.

        sigma_y0 = 360 MPa (NOT 385 MPa -- see abaqus_steel_nl_tabular(),
        which was built from a different source spreadsheet and does NOT
        match the material actually used in the original benchmark model).
        31 rows, (yield stress Pa, plastic strain) pairs taken directly --
        Abaqus *PLASTIC data IS already (stress, plastic strain), no
        eps_total -> eps_p conversion needed here.

        This is ~20-40% softer (more plastic strain at a given stress) than
        abaqus_steel_nl_tabular() through the working range 400-450 MPa --
        e.g. at 425 MPa: eps_p = 0.0014 here vs 0.00101 in the other table.
        Every benchmark run in this session prior to discovering this
        mismatch used the WRONG (385 MPa, stiffer) table and must be
        regenerated against this one before being trusted.
        """
        E_Pa = 210e9
        data = np.array([
            [360.0, 0.0],
            [385.0, 0.000263], [390.0, 0.00034],  [395.0, 0.000428],
            [400.0, 0.000531], [405.0, 0.000652], [410.0, 0.000795],
            [415.0, 0.000963], [420.0, 0.001163], [425.0, 0.0014],
            [430.0, 0.001682], [435.0, 0.002017], [440.0, 0.002416],
            [445.0, 0.002892], [450.0, 0.003458], [455.0, 0.00413],
            [460.0, 0.00493],  [465.0, 0.005878], [470.0, 0.007003],
            [475.0, 0.008336], [480.0, 0.009912], [485.0, 0.011774],
            [490.0, 0.013971], [495.0, 0.016559], [500.0, 0.019603],
            [505.0, 0.023179], [510.0, 0.027375], [515.0, 0.032289],
            [520.0, 0.038037], [525.0, 0.044752], [530.0, 0.052585],
        ])
        sigma = data[:, 0] * 1e6
        eps_p = data[:, 1]
        return cls(id=id, E=E_Pa, sigma_y0=float(sigma[0]), H=0.0,
                   eps_p_table=eps_p, sigma_y_table=sigma)

    def yield_stress(self, kap: float) -> float:
        """Current yield stress at accumulated equivalent plastic strain kap."""
        if self.eps_p_table is not None and len(self.eps_p_table) > 1:
            if kap >= self.eps_p_table[-1]:
                # Beyond table: extrapolate with H
                return float(self.sigma_y_table[-1]
                             + self.H * (kap - self.eps_p_table[-1]))
            return float(np.interp(kap, self.eps_p_table, self.sigma_y_table))
        return self.sigma_y0 + self.H * kap

    def hardening_modulus(self, kap: float) -> float:
        """Tangent plastic modulus H = d_sigma_y / d_kap at current kap."""
        if self.eps_p_table is not None and len(self.eps_p_table) > 1:
            if kap >= self.eps_p_table[-1]:
                return self.H
            i = int(np.searchsorted(self.eps_p_table, kap, side='right')) - 1
            i = max(0, min(i, len(self.eps_p_table) - 2))
            dk = self.eps_p_table[i+1] - self.eps_p_table[i]
            if dk < 1e-20:
                return self.H
            return float((self.sigma_y_table[i+1] - self.sigma_y_table[i]) / dk)
        return self.H


class PlasticState:
    """
    Per-element, per-Gauss-point, per-fibre plastic state for v4.

    Arrays
    ------
    eps_p : (n_elems, 2, n_fibres)  plastic strain per fibre
    kap   : (n_elems, 2, n_fibres)  accumulated equiv. plastic strain per fibre

    Usage
    -----
    Committed at the START of each load increment (frozen during Newton iters).
    Updated (committed) only after Newton convergence.
    Optionally carried between run_slay() calls for passage simulation.
    """
    def __init__(self, n_elems: int, n_fibres: int):
        self.n_elems  = n_elems
        self.n_fibres = n_fibres
        self.eps_p = np.zeros((n_elems, 2, n_fibres))
        self.kap   = np.zeros((n_elems, 2, n_fibres))

    def copy(self) -> 'PlasticState':
        out = PlasticState(self.n_elems, self.n_fibres)
        out.eps_p[:] = self.eps_p
        out.kap[:]   = self.kap
        return out

    def zeros_like(self) -> 'PlasticState':
        return PlasticState(self.n_elems, self.n_fibres)

    def max_eps_p(self) -> float:
        """Maximum absolute plastic strain across all elements/Gauss pts/fibres."""
        return float(np.max(np.abs(self.eps_p)))

    def max_kap(self) -> float:
        """Maximum accumulated equivalent plastic strain."""
        return float(np.max(self.kap))


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
    nodes:       List[Node]                         = field(default_factory=list)
    elements:    List[UserElement]                  = field(default_factory=list)
    sections:    List                               = field(default_factory=list)  # Section | PipeSection
    materials:   List                               = field(default_factory=list)  # Material | RambergOsgood
    joint_loads: List[JointLoad]                    = field(default_factory=list)
    line_forces: List[LineForce]                    = field(default_factory=list)
    body_forces: List[BodyForce]                    = field(default_factory=list)
    bcs:         List[BoundaryCondition]            = field(default_factory=list)
    n_steps:     int = 1


# =============================================================================
# RAMBERG-OSGOOD MATERIAL ROUTINES
# =============================================================================

def _ro_stress(eps: np.ndarray, E: float, sig_y: float,
               alpha: float, n: float,
               max_iter: int = 50, tol: float = 1e-12) -> np.ndarray:
    """
    Solve R-O equation for stress given strain, vectorised over fibres.

    R-O (DNV form):  eps/eps_y = sig/sig_y + alpha*(sig/sig_y)^n
    Solved by Newton's method on each fibre simultaneously.
    Initial guess capped at 3*sig_y to avoid overflow for large n.

    Parameters
    ----------
    eps : (nf,) fibre strains
    Returns sigma : (nf,) fibre stresses
    """
    eps_y = sig_y / E
    # Initial guess: elastic, but capped to avoid (sig/sig_y)^n overflow
    sig = np.clip(E * eps, -3.0 * sig_y, 3.0 * sig_y)
    for _ in range(max_iter):
        abs_sig_norm = np.abs(sig) / sig_y
        # Clip to prevent overflow: (sig/sig_y)^n overflows for sig >> sig_y, large n
        abs_sig_norm_c = np.minimum(abs_sig_norm, 10.0)
        # R-O residual:  f = sig/sig_y + alpha*(sig/sig_y)^n - eps/eps_y
        f  = sig / sig_y + alpha * np.sign(sig) * abs_sig_norm_c**n - eps / eps_y
        # Derivative:  df/d(sig) = 1/sig_y + alpha*n/sig_y * (|sig|/sig_y)^(n-1)
        fp = (1.0 + alpha * n * abs_sig_norm_c**(n - 1.0)) / sig_y
        dsig = f / np.where(np.abs(fp) > 1e-30, fp, 1e-30)
        sig -= dsig
        if np.max(np.abs(dsig)) < tol * sig_y:
            break
    return sig


def _ro_tangent(sig: np.ndarray, E: float, sig_y: float,
                alpha: float, n: float) -> np.ndarray:
    """
    Tangent modulus E_t = d  /d   at given stress, vectorised over fibres.

    E_t = E / (1 + alpha * n * (|sig| / sig_y)^(n-1))
    """
    abs_sig_norm = np.minimum(np.abs(sig) / sig_y, 10.0)   # clip prevents overflow
    return E / (1.0 + alpha * n * abs_sig_norm**(n - 1.0))


def _section_integrate(eps0: float, kappa: float,
                        fibre_y: np.ndarray, fibre_A: np.ndarray,
                        E: float, sig_y: float,
                        alpha: float, n: float):
    """
    Integrate fibre stresses and tangent moduli over the cross-section.

    Fibre strain:  eps_i = eps0 + y_i * kappa
    Returns
    -------
    N    : axial force
    M    : bending moment
    EA_t : axial tangent stiffness
    ES_t : coupling tangent stiffness (= 0 for symmetric section + zero eps0)
    EI_t : bending tangent stiffness
    sig  : (nf,) fibre stresses
    eps  : (nf,) fibre strains
    """
    eps_fib = eps0 + fibre_y * kappa
    sig_fib = _ro_stress(eps_fib, E, sig_y, alpha, n)
    Et_fib  = _ro_tangent(sig_fib, E, sig_y, alpha, n)

    N    = float(np.dot(sig_fib, fibre_A))
    M    = float(np.dot(sig_fib * fibre_y, fibre_A))
    EA_t = float(np.dot(Et_fib, fibre_A))
    ES_t = float(np.dot(Et_fib * fibre_y, fibre_A))
    EI_t = float(np.dot(Et_fib * fibre_y**2, fibre_A))
    return N, M, EA_t, ES_t, EI_t, sig_fib, eps_fib


# =============================================================================
# v4  INCREMENTAL PLASTICITY -- RETURN MAPPING
# =============================================================================

def _ep_return_mapping(eps_fib:     np.ndarray,
                       eps_p_old:   np.ndarray,
                       kap_old:     np.ndarray,
                       E:           float,
                       mat:         'IncrementalIsotropic'):
    """
    1D radial return mapping for an array of fibres (v4).

    Each fibre is independent -- has its own plastic strain and hardening.

    Algorithm (per fibre f):
        sigma_trial = E * (eps_fib[f] - eps_p_old[f])
        sy = mat.yield_stress(kap_old[f])
        f_trial = |sigma_trial| - sy
        if f_trial <= 0: elastic, no change
        else:
            H_cur   = mat.hardening_modulus(kap_old[f])
            d_gamma = f_trial / (E + H_cur)
            eps_p   += d_gamma * sign(sigma_trial)
            kap     += d_gamma
            sigma    = (sy + H_cur * d_gamma) * sign(sigma_trial)
            E_t      = E * H_cur / (E + H_cur)   [consistent tangent]

    Parameters
    ----------
    eps_fib  : (nf,) current total strain per fibre
    eps_p_old: (nf,) committed plastic strain (FROZEN at increment start)
    kap_old  : (nf,) committed accumulated equiv plastic strain
    E        : Young's modulus (Pa)
    mat      : IncrementalIsotropic material

    Returns
    -------
    sigma    : (nf,) updated stress
    eps_p    : (nf,) updated plastic strain
    kap      : (nf,) updated accumulated equiv plastic strain
    E_t      : (nf,) consistent tangent modulus
    """
    nf = len(eps_fib)
    eps_p = eps_p_old.copy()
    kap   = kap_old.copy()
    E_t   = np.full(nf, E)

    sigma_trial = E * (eps_fib - eps_p_old)

    # Yield stress at committed kap (vectorised via list comprehension over fibres)
    sy_arr = np.array([mat.yield_stress(float(k)) for k in kap_old])
    f_trial = np.abs(sigma_trial) - sy_arr

    plastic_mask = f_trial > 0.0
    if plastic_mask.any():
        idx = np.where(plastic_mask)[0]
        for f in idx:
            sgn    = np.sign(sigma_trial[f])
            kap0_f = float(kap_old[f])

            # ── Newton iteration for exact radial return ───────────────────
            # Finds dg ≥ 0 satisfying the consistency condition:
            #   |σ_trial - E·dg| = σ_y(κ₀ + dg)
            #
            # For linear hardening (H = const) this converges in one step.
            # For nonlinear/tabular hardening it iterates to 1 Pa tolerance.
            # The one-step formula  dg = f_trial/(E+H)  is only exact when H
            # is constant; for a tabular curve where H drops from ~65 GPa to
            # ~7 GPa the one-step version underestimates dg by up to 20%,
            # leaving the returned stress above the yield surface.
            H_f  = mat.hardening_modulus(kap0_f)
            dg   = f_trial[f] / (E + H_f)          # first-order estimate

            for _ in range(30):
                sig_rm = sigma_trial[f] - E * dg * sgn
                sy_f   = mat.yield_stress(kap0_f + dg)
                f_rm   = abs(sig_rm) - sy_f
                if abs(f_rm) < 1.0:                  # converged (1 Pa)
                    break
                H_f  = mat.hardening_modulus(kap0_f + dg)
                dg   = max(0.0, dg + f_rm / (E + H_f))

            # ── commit ─────────────────────────────────────────────────────
            eps_p[f]      += dg * sgn
            kap[f]        += dg
            sigma_trial[f] = sy_f * sgn
            E_t[f]         = E * H_f / (E + H_f) if (E + H_f) > 0 else 0.0

    return sigma_trial, eps_p, kap, E_t


def _section_integrate_ep(eps0:      float,
                           kappa:    float,
                           fibre_y:  np.ndarray,
                           fibre_A:  np.ndarray,
                           eps_p_old: np.ndarray,
                           kap_old:  np.ndarray,
                           E:        float,
                           mat:      'IncrementalIsotropic'):
    """
    Fibre section integration for incremental isotropic plasticity (v4).

    Uses the COMMITTED plastic state (eps_p_old, kap_old) from the start
    of the current increment.  The returned eps_p_new / kap_new are the
    trial-updated values at the current total strain; they are COMMITTED
    only after Newton convergence.

    Parameters
    ----------
    eps0, kappa : section centroidal strain and curvature
    fibre_y, fibre_A : fibre centroids and areas
    eps_p_old, kap_old : committed plastic state per fibre (nf,)
    E, mat      : material constants

    Returns
    -------
    N, M        : section axial force and bending moment
    EA_t, ES_t, EI_t : tangent section stiffness moduli
    sigma       : (nf,) fibre stresses
    eps_p_new   : (nf,) trial-updated plastic strains
    kap_new     : (nf,) trial-updated accumulated equiv plastic strains
    """
    eps_fib = eps0 + fibre_y * kappa
    sigma, eps_p_new, kap_new, E_t = _ep_return_mapping(
        eps_fib, eps_p_old, kap_old, E, mat)

    N    = float(np.dot(sigma, fibre_A))
    M    = float(np.dot(sigma * fibre_y, fibre_A))
    EA_t = float(np.dot(E_t, fibre_A))
    ES_t = float(np.dot(E_t * fibre_y, fibre_A))
    EI_t = float(np.dot(E_t * fibre_y**2, fibre_A))

    return N, M, EA_t, ES_t, EI_t, sigma, eps_p_new, kap_new


# =============================================================================
# =============================================================================

class MeshedStructure:
    """
    Meshed model with precomputed arrays for fast vectorised assembly.
    Key arrays (all computed once at mesh time):
      elem_dof_array : (n_elems, 6) int32 - DOF indices per element
      elem_coords    : (n_elems, 4)        - [x1,y1,x2,y2] reference coords
      elem_L0        : (n_elems,)           - reference element lengths
      elem_E/A/I     : (n_elems,)           - material/section properties
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
        """Build arrays used every NR iteration - computed once at mesh time."""
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

        # Inelastic element flags and R-O parameters
        self.elem_inelastic = np.zeros(ne, dtype=bool)
        # R-O parameters stored per element: [E, sig_y, alpha, n]
        self.elem_ro_params = np.zeros((ne, 4))
        # Fibre arrays stored as list of (y, A) - None for elastic elements
        self.elem_fibres: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * ne

        # v4: Incremental-plasticity element flags
        self.elem_ep       = np.zeros(ne, dtype=bool)   # IncrementalIsotropic + PipeSection
        self.elem_ep_mat: List[Optional['IncrementalIsotropic']] = [None] * ne  # mat ref

        for ie, (_, _, mat_id, sec_id, _) in enumerate(self.mesh_elems):
            mat = self.material_map[mat_id]
            sec = self.section_map[sec_id]
            self.elem_E[ie] = mat.E
            self.elem_A[ie] = sec.A
            self.elem_I[ie] = sec.I

            # Check for inelastic combination: RambergOsgood + PipeSection
            if isinstance(mat, RambergOsgood) and isinstance(sec, PipeSection):
                self.elem_inelastic[ie] = True
                self.elem_ro_params[ie] = [mat.E, mat.sig_y, mat.alpha, mat.n]
                fy, fA = sec.fibre_geometry()
                self.elem_fibres[ie] = (fy, fA)
            elif isinstance(mat, IncrementalIsotropic) and isinstance(sec, PipeSection):
                # v4: incremental plasticity path
                self.elem_ep[ie] = True
                self.elem_ep_mat[ie] = mat
                fy, fA = sec.fibre_geometry()
                self.elem_fibres[ie] = (fy, fA)
            elif isinstance(mat, RambergOsgood) and isinstance(sec, Section):
                # R-O with rectangular section - build fibres through depth
                self.elem_inelastic[ie] = True
                self.elem_ro_params[ie] = [mat.E, mat.sig_y, mat.alpha, mat.n]
                nf = 20
                dy_f = sec.d / nf
                fy = np.linspace(-sec.d/2 + 0.5*dy_f, sec.d/2 - 0.5*dy_f, nf)
                fA = np.full(nf, sec.b * dy_f)
                self.elem_fibres[ie] = (fy, fA)

    def dofs_of(self, node_idx: int) -> Tuple[int, int, int]:
        base = 3 * node_idx
        return base, base+1, base+2


# =============================================================================
# ELEMENT ROUTINE - used only for distributed load calculation
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
# VECTORISED ASSEMBLY - all elements processed simultaneously
# =============================================================================

def assemble(mesh: MeshedStructure,
             U: np.ndarray,
             theta_states: np.ndarray,
             dist_loads: Dict[int, Tuple[float, float]],
             joint_loads: List,
             lam: float,
             plastic_state: Optional['PlasticState'] = None
             ) -> Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray, Optional['PlasticState']]:
    """
    Assembly supporting elastic (v2), R-O inelastic (v3), and EP inelastic (v4).

    plastic_state : PlasticState | None
        If None  --> v3 path (R-O or elastic); returns ps_new = None.
        If given --> v4 EP path for IncrementalIsotropic elements.
                   Uses COMMITTED plastic state (frozen during NR iterations).
                   Returns ps_new with trial-updated plastic strains.

    Returns
    -------
    K, Fint, Fext, theta_new, ps_new
    (ps_new is None when plastic_state input is None)
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

    # ------------------------------------------------------------------
    # Geometry - all elements simultaneously
    # ------------------------------------------------------------------
    ux1 = U[dof_arr[:, 0]]; uy1 = U[dof_arr[:, 1]]; rz1 = U[dof_arr[:, 2]]
    ux2 = U[dof_arr[:, 3]]; uy2 = U[dof_arr[:, 4]]; rz2 = U[dof_arr[:, 5]]

    x1d = coords[:, 0] + ux1;  y1d = coords[:, 1] + uy1
    x2d = coords[:, 2] + ux2;  y2d = coords[:, 3] + uy2
    dx  = x2d - x1d;            dy  = y2d - y1d
    Ld  = np.hypot(dx, dy)

    theta0   = np.arctan2(coords[:, 3] - coords[:, 1],
                          coords[:, 2] - coords[:, 0])
    thetaRaw = np.arctan2(dy, dx)

    # De Souza rotation unwrap
    theta = thetaRaw.copy()
    valid = ~np.isnan(theta_states)
    if valid.any():
        d = thetaRaw[valid] - theta_states[valid]
        d -= 2*np.pi * np.round(d / (2*np.pi))
        theta[valid] = theta_states[valid] + d
    theta_new[:] = theta

    # Corotational local displacements (all elements)
    dth = theta - theta0
    u4  = Ld - L0          # axial elongation
    u3  = rz1 - dth        # relative rotation at node 1
    u6  = rz2 - dth        # relative rotation at node 2

    # ------------------------------------------------------------------
    # Separate elastic and inelastic element indices
    # ------------------------------------------------------------------
    inel_mask = mesh.elem_inelastic          # (ne,) bool  -- R-O elements
    el_mask   = ~inel_mask & ~mesh.elem_ep   # pure elastic

    # v4: prepare output plastic state (copy of input, updated in EP loop)
    ps_new: Optional[PlasticState] = None
    if plastic_state is not None:
        ps_new = plastic_state.copy()

    # Storage for Ke (ne, 6, 6) and Fint_e (ne, 6)
    Ke_all     = np.zeros((ne, 6, 6))
    Fint_e_all = np.zeros((ne, 6))

    # ------------------------------------------------------------------
    # ELASTIC ELEMENTS - fully vectorised (identical to v2)
    # ------------------------------------------------------------------
    if el_mask.any():
        ei = np.where(el_mask)[0]
        EAL = mesh.elem_E[ei] * mesh.elem_A[ei] / L0[ei]
        EIL = mesh.elem_E[ei] * mesh.elem_I[ei] / L0[ei]
        N_e  = EAL * u4[ei]
        M1_e = 4*EIL*u3[ei] + 2*EIL*u6[ei]
        M2_e = 2*EIL*u3[ei] + 4*EIL*u6[ei]

        c_e = np.cos(theta[ei]); s_e = np.sin(theta[ei])
        ne_e = len(ei)
        T1_e = np.zeros((ne_e, 3, 6))
        T1_e[:, 0, 0] = -c_e;       T1_e[:, 0, 1] = -s_e
        T1_e[:, 0, 3] =  c_e;       T1_e[:, 0, 4] =  s_e
        T1_e[:, 1, 0] = -s_e/Ld[ei]; T1_e[:, 1, 1] = c_e/Ld[ei]; T1_e[:, 1, 2] = 1.0
        T1_e[:, 1, 3] =  s_e/Ld[ei]; T1_e[:, 1, 4] =-c_e/Ld[ei]
        T1_e[:, 2, 0] = -s_e/Ld[ei]; T1_e[:, 2, 1] = c_e/Ld[ei]; T1_e[:, 2, 5] = 1.0
        T1_e[:, 2, 3] =  s_e/Ld[ei]; T1_e[:, 2, 4] =-c_e/Ld[ei]

        kl_e = np.zeros((ne_e, 3, 3))
        kl_e[:, 0, 0] = EAL
        kl_e[:, 1, 1] = 4*EIL; kl_e[:, 1, 2] = 2*EIL
        kl_e[:, 2, 1] = 2*EIL; kl_e[:, 2, 2] = 4*EIL

        tmp_e   = np.einsum('eij,ejk->eik', kl_e, T1_e)
        Ke_mat_e = np.einsum('eji,ejk->eik', T1_e, tmp_e)

        z_e = np.column_stack([s_e, -c_e, np.zeros(ne_e), -s_e,  c_e, np.zeros(ne_e)])
        r_e = np.column_stack([-c_e, -s_e, np.zeros(ne_e), c_e,  s_e, np.zeros(ne_e)])
        NL_e   = (N_e  / Ld[ei])[:, None, None]
        M12L_e = ((M1_e + M2_e) / Ld[ei])[:, None, None]
        K4_e   = (NL_e * np.einsum('ei,ej->eij', z_e, z_e)
                  + M12L_e * (np.einsum('ei,ej->eij', r_e, z_e)
                              + np.einsum('ei,ej->eij', z_e, r_e)))

        Ke_all[ei]     = Ke_mat_e + K4_e
        floc_e         = np.column_stack([N_e, M1_e, M2_e])
        Fint_e_all[ei] = np.einsum('eji,ej->ei', T1_e, floc_e)

    # ------------------------------------------------------------------
    # INELASTIC ELEMENTS - TWO-POINT GAUSS with independent point stiffness
    # ------------------------------------------------------------------
    # Proper distributed-plasticity integration. The local bending
    # stiffness (2x2 in end rotations u3,u6) and the end moments are formed
    # by integrating the section response at TWO Gauss points along the
    # element, each with its OWN tangent EI_t. This regularises strain
    # localisation: a Gauss point that yields softens only its own
    # contribution, so the element retains stiffness from the elastic
    # point and yielding cannot collapse into a single element.
    #
    # Curvature-rotation (B) functions for the cubic Hermitian beam,
    # natural coord xi in [-1,1]:
    #     kappa(xi) = B3(xi)*u3 + B6(xi)*u6
    #     B3(xi) = (3*xi - 1) / L0
    #     B6(xi) = (3*xi + 1) / L0
    # Local bending stiffness:
    #     k_bend[i,j] = (L0/2) * sum_g w_g * EI_g * Bi(xi_g) * Bj(xi_g)
    # End moments (work-conjugate to u3,u6):
    #     M_i = (L0/2) * sum_g w_g * M_g(xi_g) * Bi(xi_g)
    # where M_g is the section moment from fibre integration at xi_g.
    GAUSS_XI = np.array([-1.0/np.sqrt(3.0), +1.0/np.sqrt(3.0)])
    GAUSS_W  = np.array([1.0, 1.0])

    for ie in np.where(inel_mask)[0]:
        E_r, sig_y, alpha, n_ro = mesh.elem_ro_params[ie]
        fy, fA = mesh.elem_fibres[ie]

        L0_e = L0[ie]; Ld_e = Ld[ie]
        u4_e = u4[ie]; u3_e = u3[ie]; u6_e = u6[ie]

        eps0 = u4_e / L0_e

        # Accumulators
        kb = np.zeros((2, 2))   # bending stiffness in (u3, u6)
        M_nodal = np.zeros(2)   # work-conjugate end moments [M1, M2]
        N_g = 0.0               # axial force (averaged)
        EA_g = 0.0              # axial tangent stiffness (averaged)
        jac = L0_e / 2.0        # dx = (L0/2) dxi

        for gp in range(2):
            xi = GAUSS_XI[gp]
            w  = GAUSS_W[gp]
            B3 = (3.0*xi - 1.0) / L0_e
            B6 = (3.0*xi + 1.0) / L0_e
            kap_gp = B3*u3_e + B6*u6_e

            Ng, Mg, EAg, ESg, EIg, _, _ = _section_integrate(
                eps0, kap_gp, fy, fA, E_r, sig_y, alpha, n_ro)

            Bvec = np.array([B3, B6])
            kb      += w * jac * EIg * np.outer(Bvec, Bvec)
            M_nodal += w * jac * Mg  * Bvec
            N_g     += 0.5 * Ng       # average axial force
            EA_g    += 0.5 * EAg      # average axial stiffness

        EAL_t = EA_g / L0_e
        N_for_fint = N_g

        # Local 3x3 tangent stiffness: [axial; M1; M2]
        kl_i = np.array([
            [EAL_t, 0.0,        0.0      ],
            [0.0,   kb[0, 0],   kb[0, 1] ],
            [0.0,   kb[1, 0],   kb[1, 1] ]
        ])

        M1_for_fint = M_nodal[0]
        M2_for_fint = M_nodal[1]

        # T1 transformation matrix (3, 6)
        c_i = np.cos(theta[ie]); s_i = np.sin(theta[ie])
        T1_i = np.zeros((3, 6))
        T1_i[0, 0] = -c_i;      T1_i[0, 1] = -s_i
        T1_i[0, 3] =  c_i;      T1_i[0, 4] =  s_i
        T1_i[1, 0] = -s_i/Ld_e; T1_i[1, 1] = c_i/Ld_e; T1_i[1, 2] = 1.0
        T1_i[1, 3] =  s_i/Ld_e; T1_i[1, 4] =-c_i/Ld_e
        T1_i[2, 0] = -s_i/Ld_e; T1_i[2, 1] = c_i/Ld_e; T1_i[2, 5] = 1.0
        T1_i[2, 3] =  s_i/Ld_e; T1_i[2, 4] =-c_i/Ld_e

        Ke_mat_i = T1_i.T @ kl_i @ T1_i

        z_i = np.array([s_i, -c_i, 0.0, -s_i, c_i, 0.0])
        r_i_vec = np.array([-c_i, -s_i, 0.0, c_i, s_i, 0.0])
        K4_i = ((N_for_fint / Ld_e) * np.outer(z_i, z_i)
                + ((M1_for_fint + M2_for_fint) / Ld_e)
                  * (np.outer(r_i_vec, z_i) + np.outer(z_i, r_i_vec)))

        Ke_all[ie] = Ke_mat_i + K4_i

        floc_i = np.array([N_for_fint, M1_for_fint, M2_for_fint])
        Fint_e_all[ie] = T1_i.T @ floc_i

    # ------------------------------------------------------------------
    # v4  EP ELEMENTS -- IncrementalIsotropic + PipeSection
    # ------------------------------------------------------------------
    # Same two-point Gauss scheme as R-O, but uses _section_integrate_ep()
    # with the COMMITTED plastic state (plastic_state input, frozen per Newton iter).
    # ps_new is updated here; committed only after solve_step() convergence.
    for ie in np.where(mesh.elem_ep)[0]:
        mat_ep: IncrementalIsotropic = mesh.elem_ep_mat[ie]
        fy, fA = mesh.elem_fibres[ie]
        E_ep   = mat_ep.E

        L0_e  = L0[ie];  Ld_e = Ld[ie]
        u4_e  = u4[ie];  u3_e = u3[ie];  u6_e = u6[ie]
        eps0  = u4_e / L0_e

        kb        = np.zeros((2, 2))
        M_nodal   = np.zeros(2)
        N_g       = 0.0
        EA_g      = 0.0
        jac       = L0_e / 2.0

        for gp in range(2):
            xi  = GAUSS_XI[gp];  w = GAUSS_W[gp]
            B3  = (3.0*xi - 1.0) / L0_e
            B6  = (3.0*xi + 1.0) / L0_e
            kap_gp = B3*u3_e + B6*u6_e

            # Get committed plastic state for this element / Gauss point
            if plastic_state is not None:
                eps_p_c = plastic_state.eps_p[ie, gp, :len(fy)]
                kap_c   = plastic_state.kap  [ie, gp, :len(fy)]
            else:
                eps_p_c = np.zeros(len(fy))
                kap_c   = np.zeros(len(fy))

            (Ng, Mg, EAg, ESg, EIg,
             sig_gp, eps_p_new_gp, kap_new_gp) = _section_integrate_ep(
                eps0, kap_gp, fy, fA, eps_p_c, kap_c, E_ep, mat_ep)

            # Update ps_new with trial plastic state for this Gauss point
            if ps_new is not None:
                nf = len(fy)
                ps_new.eps_p[ie, gp, :nf] = eps_p_new_gp
                ps_new.kap  [ie, gp, :nf] = kap_new_gp

            Bvec = np.array([B3, B6])
            kb      += w * jac * EIg * np.outer(Bvec, Bvec)
            M_nodal += w * jac * Mg  * Bvec
            N_g     += 0.5 * Ng
            EA_g    += 0.5 * EAg

        EAL_t = EA_g / L0_e
        kl_i = np.array([
            [EAL_t,    0.0,       0.0      ],
            [0.0,      kb[0,0],   kb[0,1]  ],
            [0.0,      kb[1,0],   kb[1,1]  ]
        ])
        M1_f = M_nodal[0];  M2_f = M_nodal[1]

        c_i = np.cos(theta[ie]);  s_i = np.sin(theta[ie])
        T1_i = np.zeros((3, 6))
        T1_i[0,0]=-c_i; T1_i[0,1]=-s_i; T1_i[0,3]=c_i; T1_i[0,4]=s_i
        T1_i[1,0]=-s_i/Ld_e; T1_i[1,1]=c_i/Ld_e; T1_i[1,2]=1.0
        T1_i[1,3]=s_i/Ld_e;  T1_i[1,4]=-c_i/Ld_e
        T1_i[2,0]=-s_i/Ld_e; T1_i[2,1]=c_i/Ld_e; T1_i[2,5]=1.0
        T1_i[2,3]=s_i/Ld_e;  T1_i[2,4]=-c_i/Ld_e

        Ke_mat_i = T1_i.T @ kl_i @ T1_i
        z_i  = np.array([s_i, -c_i, 0., -s_i, c_i, 0.])
        r_i  = np.array([-c_i, -s_i, 0., c_i, s_i, 0.])
        K4_i = ((N_g/Ld_e) * np.outer(z_i, z_i)
                + ((M1_f+M2_f)/Ld_e) * (np.outer(r_i,z_i) + np.outer(z_i,r_i)))
        Ke_all[ie] = Ke_mat_i + K4_i

        floc_i = np.array([N_g, M1_f, M2_f])
        Fint_e_all[ie] = T1_i.T @ floc_i

    # ------------------------------------------------------------------
    # Global assembly - COO -> CSR
    # ------------------------------------------------------------------
    rows_g = np.repeat(dof_arr, 6, axis=1).reshape(ne, 36)
    cols_g = np.tile(dof_arr, (1, 6)).reshape(ne, 36)
    K = csr_matrix((Ke_all.reshape(ne, 36).ravel(),
                    (rows_g.ravel(), cols_g.ravel())),
                   shape=(ndof, ndof))

    np.add.at(Fint, dof_arr.ravel(), Fint_e_all.ravel())

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

    return K, Fint, Fext, theta_new, ps_new


# =============================================================================
# BOUNDARY CONDITIONS - penalty method, no matrix copy
# =============================================================================

def apply_bcs_sparse(K: csr_matrix,
                     Fres: np.ndarray,
                     bc_dofs: List[int],
                     bc_delta: List[float]) -> Tuple[csr_matrix, np.ndarray]:
    """
    Apply prescribed displacement increments via large-penalty method.
    Adds penalty * I to diagonal at BC DOFs and sets RHS accordingly.
    No matrix structural changes - O(1) per BC DOF.
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
               n_increments: int = 20,
               verbose: bool = False,
               plastic_state: Optional['PlasticState'] = None,
               ) -> Tuple[np.ndarray, List[dict], Optional['PlasticState']]:
    """
    Incremental Newton-Raphson with plastic state commitment (v4).

    plastic_state : PlasticState | None
        None  --> v3 path (R-O or elastic).
        Given --> v4 path: committed each converged increment, returned on exit.

    Multi-step propagation fix (v4 bug fix)
    ----------------------------------------
    THREE bugs were present in the original implementation:

    Bug A — loads: step_loads['*_prop'] (loads started in a previous step that
    continue into this one) were merged with step_loads['*_init'] (loads new
    this step) and BOTH were scaled by lam_t.  At lam_t=0 every load became
    zero, but Fint from the previous step was non-zero → residual = 0 − Fint_prev
    = a large unload force.  The solver spent all Newton iterations fighting to
    zero the structure.
    Fix: propagating loads (prop) carry at FULL amplitude (lam_t=1) throughout
    the step; only new loads (init) ramp from 0 → 1 with lam_t.

    Bug B — boundary conditions: bc_delta used bc_vals_target[k]*lam_t as the
    target.  For a BC that is already at its target from a prior step, at
    lam_t=0 this gives target=0 → bc_delta = -bc_val → a huge penalty force
    drives the node back to zero.
    Fix: record U_bc_start at the start of each step and use an incremental
    target:  target(lam_t) = U_bc_start[k] + (bc_val − U_bc_start[k]) * lam_t.
    A propagating BC (U_bc_start[k] ≈ bc_val) produces zero delta throughout;
    a new BC (U_bc_start[k] = 0) ramps correctly.

    Bug C — body forces: solve_step only included line_init/line_prop; body
    forces (self-weight in multi-step S-lay models) were silently dropped.
    The Riks solver already handled body forces correctly.
    Fix: body_init and body_prop are now included alongside line forces.

    Returns
    -------
    (U_cur, history, plastic_state_final)
    plastic_state_final is None when input is None (v3 backward compat).
    """
    history     = []
    U_cur       = U.copy()
    inc_size    = 1.0 / n_increments
    lam         = 0.0
    theta_state = np.full(mesh.n_elems, np.nan)

    # v4: committed plastic state (initialised from input or zero)
    ps_committed: Optional[PlasticState] = (plastic_state.copy()
                                             if plastic_state is not None else None)

    # Free DOF mask (exclude prescribed DOFs from convergence check)
    free_mask = np.ones(mesh.n_dofs, dtype=bool)
    for d in bc_dofs:
        free_mask[d] = False

    # ------------------------------------------------------------------
    # FIX A: Separate PROPAGATING loads (full amplitude throughout) from
    #        INITIATING loads (ramp 0 → 1 with lam_t).
    # Body forces included alongside line forces (Bug C fix).
    # ------------------------------------------------------------------
    raw_dist_init: Dict[int, Tuple[float, float]] = {}
    raw_dist_prop: Dict[int, Tuple[float, float]] = {}
    for (eidx, qx, qy) in step_loads.get('line_init', []):
        p = raw_dist_init.get(eidx, (0.0, 0.0))
        raw_dist_init[eidx] = (p[0]+qx, p[1]+qy)
    for (eidx, qx, qy) in step_loads.get('line_prop', []):
        p = raw_dist_prop.get(eidx, (0.0, 0.0))
        raw_dist_prop[eidx] = (p[0]+qx, p[1]+qy)
    for (eidx, bx, by) in step_loads.get('body_init', []):
        p = raw_dist_init.get(eidx, (0.0, 0.0))
        raw_dist_init[eidx] = (p[0]+bx, p[1]+by)
    for (eidx, bx, by) in step_loads.get('body_prop', []):
        p = raw_dist_prop.get(eidx, (0.0, 0.0))
        raw_dist_prop[eidx] = (p[0]+bx, p[1]+by)

    raw_joints_init = []
    raw_joints_prop = []
    for jl in step_loads.get('joint_init', []):
        mi = mesh.user_node_to_mesh[jl.node_id]
        raw_joints_init.append((mi, jl.Fx, jl.Fy, jl.Mz))
    for jl in step_loads.get('joint_prop', []):
        mi = mesh.user_node_to_mesh[jl.node_id]
        raw_joints_prop.append((mi, jl.Fx, jl.Fy, jl.Mz))

    # ------------------------------------------------------------------
    # FIX B: Record displacement at START of step for incremental BC
    #        targeting.  Propagating BCs (already at target) → zero delta.
    # ------------------------------------------------------------------
    U_bc_start = [float(U_cur[d]) for d in bc_dofs]

    while lam < 1.0 - 1e-10:
        lam_t = min(lam + inc_size, 1.0)

        # FIX A continued: prop loads at full amplitude, init loads scaled
        dist_scaled: Dict[int, Tuple[float, float]] = {}
        for ie, (qx, qy) in raw_dist_prop.items():
            dist_scaled[ie] = (qx, qy)
        for ie, (qx, qy) in raw_dist_init.items():
            p = dist_scaled.get(ie, (0.0, 0.0))
            dist_scaled[ie] = (p[0] + qx*lam_t, p[1] + qy*lam_t)

        jt_scaled = ([(ni, Fx, Fy, Mz) for (ni, Fx, Fy, Mz) in raw_joints_prop] +
                     [(ni, Fx*lam_t, Fy*lam_t, Mz*lam_t)
                      for (ni, Fx, Fy, Mz) in raw_joints_init])

        U_try     = U_cur.copy()
        th_try    = theta_state.copy()
        converged = False
        ps_inc_start = ps_committed.copy() if ps_committed is not None else None
        ps_trial     = ps_inc_start   # will be updated by assemble

        for it in range(max_iter):
            K, Fint, Fext, th_try, ps_trial = assemble(
                mesh, U_try, th_try, dist_scaled, jt_scaled, lam_t,
                plastic_state=ps_inc_start)   # always pass committed state

            Fres = Fext - Fint

            # FIX B: incremental BC targeting
            # target(lam_t) = U_bc_start[k] + (bc_val − U_bc_start[k]) * lam_t
            bc_delta = [
                (U_bc_start[k] + (bc_vals_target[k] - U_bc_start[k]) * lam_t)
                - U_try[bc_dofs[k]]
                for k in range(len(bc_dofs))
            ]
            K_bc, F_bc = apply_bcs_sparse(K, Fres, bc_dofs, bc_delta)

            # Convergence check on free DOFs
            ext_mag = max(float(np.max(np.abs(Fext))),
                          float(np.max(np.abs(Fint))), 1.0)
            res     = float(np.max(np.abs(Fres[free_mask]))) / ext_mag
            if it > 0 and res < tol:
                converged = True
                break

            # Sparse direct solve
            try:
                dU = spsolve(K_bc, F_bc)
            except Exception:
                break
            dU_norm = float(np.max(np.abs(dU)))
            U_norm  = max(float(np.max(np.abs(U_try))), 1.0)
            if dU_norm > 10.0 * U_norm:
                dU *= (10.0 * U_norm / dU_norm)
            U_try += dU

        if converged:
            U_cur       = U_try
            theta_state = th_try
            lam         = lam_t
            inc_size    = min(inc_size * 1.5, 0.1)
            if ps_trial is not None:
                ps_committed = ps_trial   # COMMIT plastic state
            history.append({'lambda': lam, 'iterations': it+1,
                            'residual': res, 'U': U_cur.copy()})
        else:
            inc_size *= 0.5
            if inc_size < 1e-6:
                if verbose:
                    print(f"  Warning: minimum increment size reached at lam={lam:.4f}, advancing")
                U_cur       = U_try
                theta_state = th_try
                lam         = lam_t
                if ps_trial is not None:
                    ps_committed = ps_trial
                inc_size    = 1.0 / n_increments
                history.append({'lambda': lam, 'iterations': it+1,
                                'residual': res, 'U': U_cur.copy()})

    return U_cur, history, ps_committed


# =============================================================================
# ARC-LENGTH (RIKS) SOLVER - for post-yield / snap-through paths
# =============================================================================

def solve_step_riks(mesh: MeshedStructure,
                    U: np.ndarray,
                    step_loads: dict,
                    bc_dofs: List[int],
                    bc_vals_target: List[float],
                    max_iter: int = 50,
                    tol: float = 5e-4,
                    n_increments: int = 20,
                    arc_length: float = None,
                    psi: float = 1.0,
                    verbose: bool = False) -> Tuple[np.ndarray, List[dict]]:
    """
    Cylindrical arc-length (Riks-Crisfield) solver.

    Scales both external forces AND prescribed boundary displacements by
    the load factor lambda. The arc-length constraint:

        ||dU||^2 + psi^2 * dlam^2 * ||q_ref||^2 = arc^2

    allows traversal of limit points and the R-O flat plateau where
    load-controlled NR stalls (EI_t -> 0).

    Reference: Crisfield (1981), Comput. Struct. 13, 55-62.
    """
    ndof        = mesh.n_dofs
    history     = []
    U_cur       = U.copy()
    lam         = 0.0
    theta_state = np.full(mesh.n_elems, np.nan)

    free_mask = np.ones(ndof, dtype=bool)
    for d in bc_dofs:
        free_mask[d] = False

    # Pre-build raw load dicts (unscaled)
    raw_dist: Dict[int, Tuple[float, float]] = {}
    for (eidx, qx, qy) in (step_loads.get('line_init', [])
                            + step_loads.get('line_prop', [])
                            + step_loads.get('body_init', [])
                            + step_loads.get('body_prop', [])):
        prev = raw_dist.get(eidx, (0.0, 0.0))
        raw_dist[eidx] = (prev[0]+qx, prev[1]+qy)
    raw_joints = []
    for jl in step_loads.get('joint_init', []) + step_loads.get('joint_prop', []):
        mi = mesh.user_node_to_mesh[jl.node_id]
        raw_joints.append((mi, jl.Fx, jl.Fy, jl.Mz))

    def assemble_at(U_t, th_t, lam_t):
        dist_s = {ie: (qx*lam_t, qy*lam_t) for ie,(qx,qy) in raw_dist.items()}
        jt_s   = [(ni,Fx*lam_t,Fy*lam_t,Mz*lam_t) for (ni,Fx,Fy,Mz) in raw_joints]
        return assemble(mesh, U_t, th_t, dist_s, jt_s, lam_t)

    def apply_penalty(K, rhs, lam_t):
        """Apply prescribed BCs via large-penalty, scaling target by lam_t."""
        penalty = float(K.diagonal().max()) * 1e8
        Kp = K.copy().tolil()
        rp = rhs.copy()
        for k, d in enumerate(bc_dofs):
            Kp[d, d] += penalty
            rp[d]    += penalty * (lam_t * bc_vals_target[k] - (rhs[d] if False else 0.0))
            # correct: rp[d] = penalty * lam_t * bc_vals_target[k]
            rp[d] = penalty * lam_t * bc_vals_target[k]
        return Kp.tocsr(), rp

    #        Get reference Fext at lam=1 to set q_ref and arc_length                               
    K0, _, Fext0, _ = assemble_at(U_cur, theta_state.copy(), 1.0)
    # q_ref: full external load vector at lambda=1 including BC penalty forces
    _, Fext_penalised = apply_penalty(K0, Fext0, 1.0)
    q_ref   = Fext_penalised.copy()
    q_norm2 = max(float(np.dot(q_ref, q_ref)), 1.0)

    #        Estimate initial arc_length from linear predictor if not given             
    dlam0 = 1.0 / n_increments
    if arc_length is None:
        Kp0, rhs0 = apply_penalty(K0, Fext0 * dlam0, dlam0)
        try:
            dU_lin = spsolve(Kp0, rhs0)
            arc_length = np.sqrt(float(np.dot(dU_lin, dU_lin))
                                 + psi**2 * dlam0**2 * q_norm2)
            arc_length = max(arc_length, 1e-8)
        except Exception:
            arc_length = dlam0 * np.sqrt(q_norm2)
    arc = arc_length

    if verbose:
        print(f"  Riks: arc={arc:.4e}  q_norm={np.sqrt(q_norm2):.3e}  psi={psi}")

    dU_prev  = np.zeros(ndof)
    dlam_prev = dlam0
    n_steps_done = 0
    max_steps = n_increments * 30

    while lam < 1.0 - 1e-10 and n_steps_done < max_steps:
        n_steps_done += 1

        #        Predictor: tangent solve                                                                                                                   
        K, Fint, Fext, th_try = assemble_at(U_cur, theta_state.copy(), lam)
        Fres_pred = lam * q_ref - Fint
        Kp, rhs_q = apply_penalty(K, q_ref, 1.0)   # unit load direction
        try:
            dU_t = spsolve(Kp, rhs_q)
        except Exception:
            if verbose: print(f"  Riks predictor solve failed at lam={lam:.4f}")
            break

        denom = np.sqrt(float(np.dot(dU_t, dU_t)) + psi**2 * q_norm2)
        if denom < 1e-14:
            break
        dlam = arc / denom

        # Sign control: keep advancing in same direction as previous step
        if n_steps_done > 1:
            dot = (float(np.dot(dU_t, dU_prev))
                   + psi**2 * q_norm2 * dlam * dlam_prev)
            if dot < 0:
                dlam = -dlam

        # Clamp to [0, 1]
        if lam + dlam > 1.0:
            dlam = 1.0 - lam
        if lam + dlam < 0.0:
            dlam = -lam

        dU      = dlam * dU_t
        U_try   = U_cur + dU
        lam_t   = lam + dlam
        dU_acc  = dU.copy()
        dlam_acc = dlam

        #        Corrector iterations                                                                                                                               
        converged = False
        th_try = theta_state.copy()
        for it in range(max_iter):
            K, Fint, Fext, th_try = assemble_at(U_try, th_try, lam_t)

            # Residual: scaled external load minus internal force
            Fres = lam_t * q_ref - Fint
            # BC residual folded in via penalty
            penalty = float(K.diagonal().max()) * 1e8
            Kp = K.copy().tolil()
            for k, d in enumerate(bc_dofs):
                Kp[d, d] += penalty
                Fres[d]   = penalty * (lam_t * bc_vals_target[k] - U_try[d])
            Kp = Kp.tocsr()

            # Convergence on free DOFs
            ref_mag = max(float(np.max(np.abs(Fint[free_mask]))), 1.0)
            res = float(np.max(np.abs(Fres[free_mask]))) / ref_mag
            if it > 0 and res < tol:
                converged = True
                break

            # Two corrector solves
            rhs_q2 = q_ref.copy()
            for k, d in enumerate(bc_dofs):
                rhs_q2[d] = penalty * bc_vals_target[k]
            try:
                dU_F = spsolve(Kp, Fres)
                dU_q = spsolve(Kp, rhs_q2)
            except Exception:
                break

            # Quadratic arc-length constraint: a*ddl^2 + b*ddl + c = 0
            a = (float(np.dot(dU_q, dU_q)) + psi**2 * q_norm2)
            v = dU_acc + dU_F
            b = 2.0*(float(np.dot(v, dU_q)) + psi**2 * dlam_acc * q_norm2)
            c = float(np.dot(v, v)) + psi**2 * dlam_acc**2 * q_norm2 - arc**2

            disc = b*b - 4.0*a*c
            if disc < 0.0: disc = 0.0
            sq = np.sqrt(disc)
            ddl1 = (-b + sq) / (2.0*a)
            ddl2 = (-b - sq) / (2.0*a)

            # Choose root that keeps dot product with predictor positive
            dot1 = (float(np.dot(dU_acc + dU_F + ddl1*dU_q, dU_t))
                    + psi**2 * q_norm2 * (dlam_acc + ddl1) * dlam)
            dot2 = (float(np.dot(dU_acc + dU_F + ddl2*dU_q, dU_t))
                    + psi**2 * q_norm2 * (dlam_acc + ddl2) * dlam)
            ddlam = ddl1 if dot1 >= dot2 else ddl2

            dU_corr  = dU_F + ddlam * dU_q
            U_try   += dU_corr
            lam_t   += ddlam
            dU_acc  += dU_corr
            dlam_acc += ddlam

        #        Accept or cut arc                                                                                                                                        
        if converged and 0.0 <= lam_t <= 1.0 + 1e-6:
            U_cur       = U_try.copy()
            theta_state = th_try.copy()
            lam         = min(lam_t, 1.0)
            dU_prev     = dU_acc.copy()
            dlam_prev   = dlam_acc
            history.append({'lambda': lam, 'iterations': it+1,
                            'residual': res, 'U': U_cur.copy()})
            if verbose:
                print(f"    step {n_steps_done:3d}: lam={lam:.4f}  "
                      f"it={it+1}  res={res:.2e}  arc={arc:.3e}")
            # Grow arc on easy steps
            if it < 4:
                arc = min(arc * 1.2, arc_length * 3)
        else:
            arc *= 0.5
            if verbose:
                print(f"    step {n_steps_done:3d}: FAIL lam={lam:.4f}  "
                      f"arc cut to {arc:.3e}")
            if arc < arc_length * 1e-6:
                if verbose:
                    print(f"  Riks: min arc length reached at lam={lam:.4f}")
                break

    return U_cur, history


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

class FEARunner:
    """
    High-level runner - identical API to v1, fully optimised internals.
    Drop-in replacement: change 'import nlfea' to point to this file.
    """

    def __init__(self, model: Model):
        self.model   = model
        self.mesh    = MeshedStructure(model)
        self.results = []     # list of increment history dicts per step
        self.U_steps = []     # displacement vector at end of each step
        self.U       = np.zeros(self.mesh.n_dofs)
        # v4: plastic state -- None for elastic/R-O models, PlasticState for EP
        self.plastic_state: Optional[PlasticState] = None

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
            max_iter: int = 50, tol: float = 5e-4, n_increments: int = 20,
            method: str = 'newton', arc_length: float = None, psi: float = 1.0):
        """
        Run all analysis steps.

        method : 'newton' (default) - incremental Newton-Raphson with
                 adaptive load/displacement stepping (solve_step).
                 'riks' - cylindrical arc-length solver (solve_step_riks),
                 for post-yield / snap-through paths where Newton stalls.
        arc_length, psi : passed to the Riks solver (auto if arc_length None).
        """
        t0 = time.time()
        # v4: detect n_fibres for PlasticState sizing (max across EP elements)
        mesh = self.mesh
        has_ep = mesh.elem_ep.any()
        if has_ep:
            n_fib_max = max(
                len(mesh.elem_fibres[ie][0])
                for ie in range(mesh.n_elems) if mesh.elem_ep[ie]
            )
            if self.plastic_state is None:
                self.plastic_state = PlasticState(mesh.n_elems, n_fib_max)

        for step in range(1, self.model.n_steps + 1):
            if verbose:
                print(f"\n--- Step {step} ({method}) ---")
            sl               = self._get_step_loads(step)
            bc_dofs, bc_vals = self._get_bc_info(step)
            if method == 'riks':
                U_new, hist = solve_step_riks(self.mesh, self.U, sl,
                                              bc_dofs, bc_vals,
                                              max_iter=max_iter, tol=tol,
                                              n_increments=n_increments,
                                              arc_length=arc_length, psi=psi,
                                              verbose=verbose)
                ps_new = self.plastic_state   # Riks not updated for EP yet
            else:
                U_new, hist, ps_new = solve_step(
                    self.mesh, self.U, sl, bc_dofs, bc_vals,
                    max_iter=max_iter, tol=tol,
                    n_increments=n_increments,
                    verbose=verbose,
                    plastic_state=self.plastic_state)
            self.U = U_new
            self.U_steps.append(U_new.copy())
            if ps_new is not None:
                self.plastic_state = ps_new
            self.results.append(hist)
            if verbose and hist:
                info = (f"  {len(hist)} increments | residual {hist[-1]['residual']:.2e}"
                        f" | lambda {hist[-1]['lambda']:.4f}")
                if ps_new is not None:
                    info += f" | max_eps_p {ps_new.max_eps_p()*100:.4f}%"
                print(info)
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

    def get_element_strains(self, elem_id: int, step: int = -1) -> dict:
        """
        Return strain and stress state at extreme fibres for a user element.

        For inelastic (RO + PipeSection) elements: uses fibre integration.
        For elastic elements: uses Euler-Bernoulli linear strain distribution.

        Returns
        -------
        dict with keys:
            eps_axial : float  membrane (axial) strain at section centroid
            kappa     : float  curvature (1/m)
            eps_top   : float  extreme fibre strain at +y (tension side)
            eps_bot   : float  extreme fibre strain at -y (compression side)
            eps_max   : float  max(|eps_top|, |eps_bot|)
            sigma_top : float  extreme fibre stress (Pa) - NaN for elastic elements
            sigma_bot : float  extreme fibre stress (Pa) - NaN for elastic elements
            elem_ids  : list   mesh element indices for this user element
        """
        U = self.U_steps[step] if self.U_steps else self.U
        mesh = self.mesh

        mesh_indices = mesh.user_elem_to_mesh.get(elem_id, [])
        if not mesh_indices:
            raise ValueError(f"User element {elem_id} not found in mesh.")

        # Report worst (max |eps|) across all sub-elements
        eps_axial_max = 0.0; kappa_max = 0.0
        eps_top_max = 0.0; eps_bot_max = 0.0
        sig_top = float('nan'); sig_bot = float('nan')
        worst_eps = 0.0

        for ie in mesh_indices:
            dofs = mesh.elem_dof_array[ie]
            ux1, uy1, rz1 = U[dofs[0]], U[dofs[1]], U[dofs[2]]
            ux2, uy2, rz2 = U[dofs[3]], U[dofs[4]], U[dofs[5]]

            coords = mesh.elem_coords[ie]
            L0_e   = mesh.elem_L0[ie]

            x1d = coords[0] + ux1; y1d = coords[1] + uy1
            x2d = coords[2] + ux2; y2d = coords[3] + uy2
            Ld_e = float(np.hypot(x2d - x1d, y2d - y1d))

            theta0_e   = float(np.arctan2(coords[3]-coords[1], coords[2]-coords[0]))
            theta_e    = float(np.arctan2(y2d - y1d, x2d - x1d))
            dth_e      = theta_e - theta0_e
            u4_e = Ld_e - L0_e
            u3_e = rz1 - dth_e
            u6_e = rz2 - dth_e

            eps0_e  = u4_e / L0_e
            kappa_e = (u6_e - u3_e) / L0_e   # correct: (u6-u3)/L0

            sec_id = mesh.mesh_elems[ie][3]
            sec    = mesh.section_map[sec_id]

            if mesh.elem_inelastic[ie]:
                # R-O: Use fibre integration for extreme fibre values.
                # r_o taken directly from the section's true outer radius
                # (sec.r_o = D_o/2), NOT reconstructed from fibre spacing.
                # v4.1 fix: the old reconstruction (max|fy| + 0.5*(fy[1]-fy[0]))
                # is exact ONLY for the Cartesian fibre model (evenly-spaced
                # strips span exactly to r_o by construction). For the polar
                # (angular) model, fibre y-values are r_mid*sin(theta) --
                # never reaching r_o and not evenly spaced -- so the old
                # formula silently returned ~1.28x the true r_o (verified
                # numerically: 260.8mm reconstructed vs 203.2mm true r_o for
                # a 16in x 21mm pipe), inflating every reported strain/stress
                # for polar-section elements by ~28%. sec.r_o is correct and
                # section-type-agnostic; for the Cartesian case it reproduces
                # the old value exactly (no change to any previously
                # validated fibre-model result).
                fy, fA = mesh.elem_fibres[ie]
                E_r, sig_y, alpha, n_ro = mesh.elem_ro_params[ie]
                r_o  = float(sec.r_o)

                et = eps0_e + r_o * kappa_e    # top fibre (+y)
                eb = eps0_e - r_o * kappa_e    # bottom fibre (-y)
                st = float(_ro_stress(np.array([et]), E_r, sig_y, alpha, n_ro)[0])
                sb = float(_ro_stress(np.array([eb]), E_r, sig_y, alpha, n_ro)[0])

            elif mesh.elem_ep[ie]:
                # v4 EP: use return mapping with committed plastic state.
                # r_o fix: see note in the elem_inelastic branch above --
                # same bug, same fix, same reasoning.
                fy, fA = mesh.elem_fibres[ie]
                mat_ep = mesh.elem_ep_mat[ie]
                r_o    = float(sec.r_o)

                et = eps0_e + r_o * kappa_e
                eb = eps0_e - r_o * kappa_e

                # Use plastic state if available (from runner)
                if (hasattr(self, 'plastic_state') and
                        self.plastic_state is not None):
                    nf = len(fy)
                    # Use Gauss point 0 as representative for extreme-fibre estimate
                    ep_c = self.plastic_state.eps_p[ie, 0, :nf]
                    ka_c = self.plastic_state.kap  [ie, 0, :nf]
                    # Evaluate at top and bottom fibres
                    st_arr, *_ = _ep_return_mapping(
                        np.array([et]), np.array([ep_c[0]]),
                        np.array([ka_c[0]]), mat_ep.E, mat_ep)
                    sb_arr, *_ = _ep_return_mapping(
                        np.array([eb]), np.array([ep_c[-1]]),
                        np.array([ka_c[-1]]), mat_ep.E, mat_ep)
                    st = float(st_arr[0]);  sb = float(sb_arr[0])
                else:
                    st = mat_ep.E * et;  sb = mat_ep.E * eb
            else:
                r_o = float(getattr(sec, 'd', getattr(sec, 'D_o', 0.0))) / 2.0
                et = eps0_e + r_o * kappa_e
                eb = eps0_e - r_o * kappa_e
                E_e = mesh.elem_E[ie]
                st  = E_e * et
                sb  = E_e * eb

            cand = max(abs(et), abs(eb))
            if cand >= worst_eps:
                worst_eps    = cand
                eps_axial_max = eps0_e
                kappa_max    = kappa_e
                eps_top_max  = et
                eps_bot_max  = eb
                sig_top      = st
                sig_bot      = sb

        return {
            'eps_axial':  eps_axial_max,
            'kappa':      kappa_max,
            'eps_top':    eps_top_max,
            'eps_bot':    eps_bot_max,
            'eps_max':    max(abs(eps_top_max), abs(eps_bot_max)),
            'sigma_top':  sig_top,
            'sigma_bot':  sig_bot,
            'elem_ids':   mesh_indices,
        }

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
