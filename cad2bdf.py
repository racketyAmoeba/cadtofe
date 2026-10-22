#!/usr/bin/env python3
"""
cad2bdf - CAD (STEP / IGES / BREP) -> Nastran BDF, with mesh quality ENFORCED.

The contract: a BDF is only written as "certified" if the *written file itself*, re-read by an
independent parser, satisfies every criterion in your config. If the criteria cannot be met
the tool says so, names the offending elements, and exits non-zero.

    python cad2bdf.py init-config > my.toml        # template with every option documented
    python cad2bdf.py mesh  part.step -c my.toml -o part.bdf
    python cad2bdf.py check existing.bdf -c my.toml   # audit ANY BDF against the same criteria
    python cad2bdf.py selftest                         # verify the install (uses real gmsh if present)

Needs: Python >= 3.11, numpy, gmsh (pip install gmsh).  Optional: scipy (faster node merge).
Exit codes: 0 certified | 1 error | 2 criteria not met.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("cad2bdf needs Python >= 3.11 (for tomllib).")

__version__ = "0.1.0"

# --------------------------------------------------------------------------------------
# Element tables
# --------------------------------------------------------------------------------------
ORDER = ["tria3", "quad4", "tria6", "quad8", "tet4", "tet10"]
KIND = {"tria3": "tria", "tria6": "tria", "quad4": "quad", "quad8": "quad", "tet4": "tet", "tet10": "tet"}
NCORNER = {"tria3": 3, "tria6": 3, "quad4": 4, "quad8": 4, "tet4": 4, "tet10": 4}
NNODES = {"tria3": 3, "quad4": 4, "tria6": 6, "quad8": 8, "tet4": 4, "tet10": 10}
SHELLS = ("tria3", "quad4", "tria6", "quad8")
CARD = {"tria3": "CTRIA3", "quad4": "CQUAD4", "tria6": "CTRIA6", "quad8": "CQUAD8",
        "tet4": "CTETRA", "tet10": "CTETRA"}
# Nastran mid-side node k (stored at index NCORNER+k) lies on this corner pair.
MIDS = {"tria6": [(0, 1), (1, 2), (2, 0)],
        "quad8": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "tet10": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]}
# Reversing element normal: new local order expressed as old local indices.
FLIP = {"tria3": [0, 2, 1], "quad4": [0, 3, 2, 1],
        "tria6": [0, 2, 1, 5, 4, 3], "quad8": [0, 3, 2, 1, 7, 6, 5, 4]}
GMSH_TYPE = {2: "tria3", 3: "quad4", 4: "tet4", 9: "tria6", 10: "quad8", 11: "tet10", 16: "quad8"}
GMSH_NODES = {10: 9}                                     # gmsh quad9 -> drop centre node
GMSH_PERM = {"tet10": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]}    # gmsh tet10 -> Nastran CTETRA ordering

METRIC_NAMES = {
    "tria": ["min_angle", "max_angle", "aspect", "skew", "area", "edge_min", "edge_max"],
    "quad": ["min_angle", "max_angle", "aspect", "skew", "taper", "warp", "jacobian",
             "area", "edge_min", "edge_max"],
    "tet": ["aspect", "collapse", "min_dihedral", "max_dihedral", "quality", "volume",
            "edge_min", "edge_max"],
}
for _k in METRIC_NAMES:
    METRIC_NAMES[_k].append("midnode_dev")   # only evaluated for 2nd-order elements


class CadError(Exception):
    """Fatal: bad input or configuration. Never retried."""


class MeshingFailure(CadError):
    """The mesher failed on one attempt. The retry loop may try a different strategy."""


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
DEFAULT_TOML = r'''# cad2bdf configuration.  Every key shown is the default; delete what you don't change.
# Units = the CAD file's units (mm for most STEP files); the BDF is written in the same units.

[mesh]
type = "shell"             # "shell": surface/midsurface geometry -> CTRIA3/CQUAD4 (or CTRIA6/CQUAD8)
                           # "tet"  : solid geometry -> CTETRA (4 or 10 node)
order = 1                  # 1 = linear, 2 = quadratic (mid-side nodes placed on the CAD geometry)
target_size = 5.0          # maximum element edge length
min_size = 0.5             # floor for local refinement; features smaller than this stay unresolved
curvature_elements = 12    # minimum elements per full circle on curved geometry (0 = off)
quad_mode = "none"         # shell only: "none" = all tria | "dominant" = quad-dominant | "all" = all quad
use_skin = false           # shell type + SOLID input: mesh the solid's outer faces (NOT a midsurface!)
length_scale = 1.0         # multiply coordinates on output (e.g. 1000 for m -> mm)

[cad]
heal = true                # OpenCASCADE shape healing: sew faces, fix small edges/faces, make solids
heal_tolerance = 1e-3
conformal = false          # fragment touching parts so interfaces share nodes (assemblies)
target_unit = "MM"         # convert imported geometry to this unit ("" = leave as imported)

[cleanup]
merge_tolerance = 1e-5     # nodes closer than this are merged (0 = only report them)
fix_orientation = true     # shells: consistent normals across each connected set of faces

[iterate]
max_iterations = 8
shrink = 0.9               # global size factor per retry (compounds: 0.9, 0.81, ...)
local_refine = true        # additionally refine around elements that violate a criterion

[checks]                   # hard requirements that aren't per-element metrics
orientation = true         # shell normals consistent across shared edges
duplicate_nodes = true     # no coincident nodes after cleanup
max_free_edges = -1        # shells: maximum allowed free edges (-1 = report only)

[output]
format = "large"           # "large" (16-char GRID fields, full precision) | "small" (8-char) | "free"
deck = "full"              # "full" = SOL/CEND/BEGIN BULK/ENDDATA | "bulk" = bulk entries only (INCLUDE)
title = "cad2bdf model"

# ---- Per-element quality criteria -------------------------------------------------------
# Each metric takes min and/or max (inclusive). Unlisted metrics are NOT enforced. Names are
# validated: a typo is an error, never a silently skipped check. If you define a [quality.*]
# table it REPLACES the default for that element kind, so what you write is exactly what is enforced.
# NOTE: these values are generic placeholders - replace them with your program's criteria.
#
# tria: min_angle max_angle aspect skew area edge_min edge_max
# quad: min_angle max_angle aspect skew taper warp jacobian area edge_min edge_max
# tet : aspect collapse min_dihedral max_dihedral quality volume edge_min edge_max
# any 2nd-order kind additionally: midnode_dev (mid-node offset from edge midpoint / edge length)

[quality.tria]
min_angle = { min = 30.0 }
max_angle = { max = 120.0 }
aspect    = { max = 3.0 }
skew      = { max = 50.0 }
midnode_dev = { max = 0.2 }

[quality.quad]
min_angle = { min = 45.0 }
max_angle = { max = 135.0 }
aspect    = { max = 3.0 }
skew      = { max = 45.0 }
taper     = { max = 0.5 }
warp      = { max = 0.05 }
jacobian  = { min = 0.6 }
midnode_dev = { max = 0.2 }

[quality.tet]
aspect       = { max = 5.0 }
collapse     = { min = 0.1 }
min_dihedral = { min = 15.0 }
max_dihedral = { max = 150.0 }
quality      = { min = 0.2 }
midnode_dev  = { max = 0.2 }

# ---- Materials and properties -----------------------------------------------------------
# Units below are tonne-mm-s (E in MPa, density in tonne/mm^3).  PLACEHOLDER values: the tool
# warns loudly while these defaults are in use.  Rules are matched in order against the CAD
# part name (STEP label, else SURF_<tag> / VOL_<tag>) with re.search; the first match wins.
# Parts matching no rule are an error. Defining any [[property]] replaces this default.

[[material]]
id = 1
name = "PLACEHOLDER_AL"
E = 71700.0
nu = 0.33
rho = 2.81e-9

[[property]]
match = ".*"
material = 1
thickness = 1.0            # shells only
placeholder = true
'''


def deep_merge(base: dict, over: dict, replace: tuple[str, ...] = ("quality", "material", "property")) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k in replace:
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                merged = dict(base[k])
                merged.update(v)              # per element kind: user table replaces default table
                out[k] = merged
            else:
                out[k] = v
        elif isinstance(v, dict) and isinstance(base.get(k), dict):
            out[k] = deep_merge(base[k], v, ())
        else:
            out[k] = v
    return out


def load_config(path: str | None) -> dict:
    cfg = tomllib.loads(DEFAULT_TOML)
    if path:
        with open(path, "rb") as fh:
            cfg = deep_merge(cfg, tomllib.load(fh))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    m = cfg["mesh"]
    if m["type"] not in ("shell", "tet"):
        raise CadError("mesh.type must be 'shell' or 'tet'")
    if m["order"] not in (1, 2):
        raise CadError("mesh.order must be 1 or 2")
    if m["quad_mode"] not in ("none", "dominant", "all"):
        raise CadError("mesh.quad_mode must be none | dominant | all")
    if m["min_size"] > m["target_size"]:
        raise CadError("mesh.min_size must be <= mesh.target_size")
    if cfg["output"]["format"] not in ("large", "small", "free"):
        raise CadError("output.format must be large | small | free")
    if cfg["output"]["deck"] not in ("full", "bulk"):
        raise CadError("output.deck must be full | bulk")
    for kind, table in cfg["quality"].items():
        if kind not in METRIC_NAMES:
            raise CadError(f"[quality.{kind}]: unknown element kind (use tria | quad | tet)")
        for name, lim in table.items():
            if name not in METRIC_NAMES[kind]:
                raise CadError(f"[quality.{kind}] unknown metric '{name}'. Valid: {', '.join(METRIC_NAMES[kind])}")
            if not isinstance(lim, dict) or not set(lim) <= {"min", "max"} or not lim:
                raise CadError(f"[quality.{kind}] {name} must look like {{ min = 1.0, max = 2.0 }}")
    mats = {x["id"] for x in cfg["material"]}
    for r in cfg["property"]:
        if r["material"] not in mats:
            raise CadError(f"property '{r['match']}' references unknown material {r['material']}")
        if m["type"] == "shell" and "thickness" not in r:
            raise CadError(f"property '{r['match']}': shell meshes need a thickness")


# --------------------------------------------------------------------------------------
# Mesh container + cleanup
# --------------------------------------------------------------------------------------
@dataclass
class Mesh:
    nid: np.ndarray
    xyz: np.ndarray
    conn: dict = field(default_factory=dict)   # type -> (E,k) 0-based indices into xyz
    eid: dict = field(default_factory=dict)
    pid: dict = field(default_factory=dict)

    def types(self):
        return [t for t in ORDER if t in self.conn and len(self.conn[t])]

    def n_elem(self):
        return sum(len(self.conn[t]) for t in self.types())


def compact(mesh: Mesh) -> Mesh:
    ts = mesh.types()
    if not ts:
        raise CadError("mesh contains no elements")
    used = np.unique(np.concatenate([mesh.conn[t].ravel() for t in ts]))
    remap = np.full(len(mesh.xyz), -1, np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.nid[used], mesh.xyz[used], {t: remap[mesh.conn[t]] for t in ts},
                {t: mesh.eid[t] for t in ts}, {t: mesh.pid[t] for t in ts})


def renumber(mesh: Mesh) -> Mesh:
    mesh.nid = np.arange(1, len(mesh.xyz) + 1, dtype=np.int64)
    n = 1
    for t in mesh.types():
        k = len(mesh.conn[t])
        mesh.eid[t] = np.arange(n, n + k, dtype=np.int64)
        n += k
    return mesh


def coincident_labels(xyz: np.ndarray, tol: float) -> np.ndarray:
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree
        pairs = cKDTree(xyz).query_pairs(tol, output_type="ndarray")
        n = len(xyz)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        return connected_components(g, directed=False)[1]
    except ImportError:   # coarse fallback: grid quantisation (can miss pairs straddling a cell wall)
        q = np.round(xyz / tol).astype(np.int64)
        return np.unique(q, axis=0, return_inverse=True)[1].reshape(-1)


def merge_nodes(mesh: Mesh, tol: float) -> Mesh:
    labels = coincident_labels(mesh.xyz, tol)
    ng = int(labels.max()) + 1
    if ng == len(mesh.xyz):
        return mesh
    rep = np.full(ng, len(mesh.xyz), np.int64)
    np.minimum.at(rep, labels, np.arange(len(mesh.xyz)))
    remap = rep[labels]
    return compact(Mesh(mesh.nid, mesh.xyz, {t: remap[c] for t, c in mesh.conn.items()},
                        mesh.eid, mesh.pid))


def shell_edges(mesh: Mesh):
    """Directed corner edges of all shell elements: (a, b, element_global_index), per-type offsets."""
    A, B, E, starts, off = [], [], [], {}, 0
    for t in SHELLS:
        if t not in mesh.conn or not len(mesh.conn[t]):
            continue
        c, n = mesh.conn[t][:, :NCORNER[t]], NCORNER[t]
        starts[t] = off
        for i in range(n):
            A.append(c[:, i]); B.append(c[:, (i + 1) % n]); E.append(np.arange(len(c)) + off)
        off += len(c)
    if not A:
        return None
    return np.concatenate(A), np.concatenate(B), np.concatenate(E), starts, off


def edge_pairs(a, b, e, nnode):
    """Group directed edges by undirected edge. Returns counts and (e1, e2, same_direction) for
    edges shared by exactly two elements."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * (nnode + 1) + hi
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    sel = np.nonzero(cnt[inv] == 2)[0]
    order = sel[np.argsort(inv[sel], kind="stable")]
    i1, i2 = order[0::2], order[1::2]
    same = (a[i1] < b[i1]) == (a[i2] < b[i2])
    return cnt, e[i1], e[i2], same


def fix_orientation(mesh: Mesh) -> int:
    """Make shell normals consistent over each manifold-connected patch. Returns #flipped."""
    se = shell_edges(mesh)
    if se is None:
        return 0
    a, b, e, starts, n = se
    _, e1, e2, same = edge_pairs(a, b, e, len(mesh.xyz))
    src = np.concatenate([e1, e2]); dst = np.concatenate([e2, e1]); w = np.concatenate([same, same])
    o = np.argsort(src, kind="stable")
    dst, w = dst[o].tolist(), w[o].astype(np.int8).tolist()
    ptr = np.searchsorted(src[o], np.arange(n + 1)).tolist()
    flip = [-1] * n
    for s in range(n):
        if flip[s] != -1:
            continue
        flip[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for k in range(ptr[u], ptr[u + 1]):
                v = dst[k]
                if flip[v] == -1:
                    flip[v] = flip[u] ^ w[k]
                    dq.append(v)
    flip = np.array(flip, bool)
    for t, s0 in starts.items():
        m = flip[s0:s0 + len(mesh.conn[t])]
        if m.any():
            mesh.conn[t][m] = mesh.conn[t][m][:, FLIP[t]]
    return int(flip.sum())


def cleanup(mesh: Mesh, cfg: dict, say=lambda s: None) -> Mesh:
    mesh = compact(mesh)
    tol = cfg["cleanup"]["merge_tolerance"]
    if tol > 0:
        n0 = len(mesh.xyz)
        mesh = merge_nodes(mesh, tol)
        if len(mesh.xyz) != n0:
            say(f"    merged {n0 - len(mesh.xyz)} coincident nodes")
    if cfg["cleanup"]["fix_orientation"]:
        k = fix_orientation(mesh)
        if k:
            say(f"    re-oriented {k} shell elements for consistent normals")
    return renumber(mesh)


# --------------------------------------------------------------------------------------
# Quality metrics (all vectorised).  Definitions are written out so you can compare them with
# your own standard; if yours differs, each is a few lines to change.
# --------------------------------------------------------------------------------------
def _norm(v): return np.linalg.norm(v, axis=-1)
def _dot(a, b): return np.einsum("...i,...i->...", a, b)


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(_dot(a, b) / (_norm(a) * _norm(b)), -1, 1)))


def tri_metrics(P):
    """aspect = longest edge / shortest altitude (equilateral = 1.155).
    skew = 90deg - smallest acute angle between a median and the edge it meets (equilateral = 0)."""
    p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
    e0, e1, e2 = p1 - p0, p2 - p1, p0 - p2
    L = np.stack([_norm(e0), _norm(e1), _norm(e2)], 1)
    area = 0.5 * _norm(np.cross(e0, -e2))
    h = 2 * area[:, None] / L
    ang = np.stack([_angle(e0, -e2), _angle(e1, -e0), _angle(e2, -e1)], 1)
    med = [0.5 * (p1 + p2) - p0, 0.5 * (p2 + p0) - p1, 0.5 * (p0 + p1) - p2]
    opp = [e1, e2, e0]
    th = np.stack([np.degrees(np.arccos(np.clip(np.abs(_dot(m, o)) / (_norm(m) * _norm(o)), 0, 1)))
                   for m, o in zip(med, opp)], 1)
    return dict(min_angle=ang.min(1), max_angle=ang.max(1), aspect=L.max(1) / h.min(1),
                skew=90 - th.min(1), area=area, edge_min=L.min(1), edge_max=L.max(1))


def quad_metrics(P):
    """aspect = longer / shorter bimedian. skew = 90deg - angle between bimedians.
    taper = max |corner-triangle area - mean| / mean.  jacobian = min / max corner-triangle area
    (signed, so concave/inverted corners go negative).  warp = distance between the two
    diagonals / mean diagonal length."""
    d1, d2 = P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]
    nrm = np.cross(d1, d2)
    nl = _norm(nrm)
    n = nrm / nl[:, None]
    s = np.stack([0.5 * _dot(np.cross(P[:, i] - P[:, i - 1], P[:, (i + 1) % 4] - P[:, i]), n)
                  for i in range(4)], 1)
    ang = np.stack([_angle(P[:, i - 1] - P[:, i], P[:, (i + 1) % 4] - P[:, i]) for i in range(4)], 1)
    L = np.stack([_norm(P[:, (i + 1) % 4] - P[:, i]) for i in range(4)], 1)
    m1 = 0.5 * (P[:, 2] + P[:, 3]) - 0.5 * (P[:, 0] + P[:, 1])
    m2 = 0.5 * (P[:, 3] + P[:, 0]) - 0.5 * (P[:, 1] + P[:, 2])
    l1, l2 = _norm(m1), _norm(m2)
    phi = np.degrees(np.arccos(np.clip(np.abs(_dot(m1, m2)) / (l1 * l2), 0, 1)))
    mean = s.mean(1)
    return dict(min_angle=ang.min(1), max_angle=ang.max(1),
                aspect=np.maximum(l1, l2) / np.minimum(l1, l2), skew=90 - phi,
                taper=np.abs(s - mean[:, None]).max(1) / mean, jacobian=s.min(1) / s.max(1),
                warp=np.abs(_dot(P[:, 1] - P[:, 0], n)) / (0.5 * (_norm(d1) + _norm(d2))),
                area=0.5 * nl, edge_min=L.min(1), edge_max=L.max(1))


def tet_metrics(P):
    """aspect = longest edge / shortest altitude (regular = 1.225). collapse = min over vertices of
    (altitude / sqrt(opposite face area)), signed (regular = 1.24).  quality = 6*sqrt(2)*V / L_rms^3
    (regular = 1, inverted < 0).  Dihedral angles from outward face normals."""
    a, b, c = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]
    vol = _dot(np.cross(a, b), c) / 6
    N, A = [], []
    for i in range(4):
        j, k, l = [q for q in range(4) if q != i]
        n = np.cross(P[:, k] - P[:, j], P[:, l] - P[:, j])
        n = n * np.where(_dot(n, P[:, i] - P[:, j]) > 0, -1.0, 1.0)[:, None]   # outward
        N.append(n); A.append(0.5 * _norm(n))
    A = np.stack(A, 1)
    h = 3 * vol[:, None] / A
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    L = np.stack([_norm(P[:, i] - P[:, j]) for i, j in pairs], 1)
    dih = np.degrees(np.arccos(np.clip(
        np.stack([-_dot(N[i], N[j]) / (_norm(N[i]) * _norm(N[j])) for i, j in pairs], 1), -1, 1)))
    lrms = np.sqrt((L ** 2).mean(1))
    return dict(aspect=L.max(1) / np.abs(h).min(1), collapse=(h / np.sqrt(A)).min(1),
                min_dihedral=dih.min(1), max_dihedral=dih.max(1),
                quality=6 * np.sqrt(2) * vol / lrms ** 3, volume=vol,
                edge_min=L.min(1), edge_max=L.max(1))


def midnode_dev(X, t):
    nc = NCORNER[t]
    return np.max(np.stack([_norm(X[:, nc + k] - 0.5 * (X[:, i] + X[:, j])) / _norm(X[:, j] - X[:, i])
                            for k, (i, j) in enumerate(MIDS[t])], 1), 1)


def compute_metrics(mesh: Mesh) -> dict:
    out = {}
    fn = {"tria": tri_metrics, "quad": quad_metrics, "tet": tet_metrics}
    with np.errstate(all="ignore"):
        for t in mesh.types():
            X = mesh.xyz[mesh.conn[t]]
            m = fn[KIND[t]](X[:, :NCORNER[t]])
            if t in MIDS:
                m["midnode_dev"] = midnode_dev(X, t)
            out[t] = m
    return out


def midside_order_errors(mesh: Mesh) -> int:
    """Elements whose mid-side nodes are not nearest to the corner-pair midpoint Nastran expects.
    Catches a wrong gmsh->Nastran node permutation or a corrupt BDF."""
    bad = 0
    with np.errstate(all="ignore"):
        for t, mids in MIDS.items():
            if t not in mesh.conn or not len(mesh.conn[t]):
                continue
            X, nc = mesh.xyz[mesh.conn[t]], NCORNER[t]
            pairs = [(i, j) for i in range(nc) for j in range(i + 1, nc)]
            mp = np.stack([0.5 * (X[:, i] + X[:, j]) for i, j in pairs], 1)
            wrong = np.zeros(len(X), bool)
            for k, (i, j) in enumerate(mids):
                wrong |= _norm(mp - X[:, nc + k][:, None, :]).argmin(1) != pairs.index((min(i, j), max(i, j)))
            bad += int(wrong.sum())
    return bad


# --------------------------------------------------------------------------------------
# Evaluation against criteria
# --------------------------------------------------------------------------------------
@dataclass
class Report:
    ok: bool
    hard: list
    n_bad: int
    stats: dict
    topology: dict
    rows: list          # (etype, eid, pid, metric, value, min, max, cx, cy, cz)
    bad_info: list      # (centroids (m,3), local size (m,)) per type, for refinement
    counts: dict

    @property
    def score(self):
        return (len(self.hard), self.n_bad)


def evaluate(mesh: Mesh, cfg: dict) -> Report:
    metrics = compute_metrics(mesh)
    rows, stats, bad_info, hard, n_bad = [], {}, [], [], 0
    counts = {t: int(len(mesh.conn[t])) for t in mesh.types()}
    for t, m in metrics.items():
        crit = cfg["quality"].get(KIND[t], {})
        X = mesh.xyz[mesh.conn[t][:, :NCORNER[t]]]
        cen = X.mean(1)
        bad_any = np.zeros(len(cen), bool)
        stats[t] = {}
        for name, v in m.items():
            lim = crit.get(name, {})
            lo, hi = lim.get("min"), lim.get("max")
            bad = np.zeros(len(v), bool)
            if lo is not None:
                bad |= ~(v >= lo)          # NaN counts as a violation
            if hi is not None:
                bad |= ~(v <= hi)
            f = v[np.isfinite(v)]
            stats[t][name] = dict(min=float(f.min()) if f.size else None, mean=float(f.mean()) if f.size else None,
                                  max=float(f.max()) if f.size else None, fails=int(bad.sum()),
                                  min_limit=lo, max_limit=hi)
            for i in np.nonzero(bad)[0]:
                rows.append((t, int(mesh.eid[t][i]), int(mesh.pid[t][i]), name, float(v[i]), lo, hi,
                             *map(float, cen[i])))
            bad_any |= bad
        n_bad += int(bad_any.sum())
        if bad_any.any():
            bad_info.append((cen[bad_any], np.nan_to_num(m["edge_max"][bad_any], nan=0.0)))

    # structural integrity (not metric-based; always fatal)
    topo = {}
    rep_nodes = sum(int((np.sort(c, 1)[:, 1:] == np.sort(c, 1)[:, :-1]).any(1).sum()) for c in mesh.conn.values())
    topo["elements_with_repeated_nodes"] = rep_nodes
    if rep_nodes:
        hard.append(f"{rep_nodes} elements reference the same node twice")
    topo["midside_order_errors"] = midside_order_errors(mesh)
    if topo["midside_order_errors"]:
        hard.append(f"{topo['midside_order_errors']} higher-order elements have mis-ordered mid-side nodes")
    tol = cfg["cleanup"]["merge_tolerance"] or 1e-6
    topo["duplicate_nodes"] = int(len(mesh.xyz) - (coincident_labels(mesh.xyz, tol).max() + 1)) if len(mesh.xyz) else 0
    if cfg["checks"]["duplicate_nodes"] and topo["duplicate_nodes"]:
        hard.append(f"{topo['duplicate_nodes']} coincident nodes (tol {tol:g})")
    se = shell_edges(mesh)
    if se is not None:
        a, b, e, _, _ = se
        cnt, _, _, same = edge_pairs(a, b, e, len(mesh.xyz))
        topo.update(free_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
                    inconsistent_orientation_edges=int(same.sum()))
        if cfg["checks"]["orientation"] and topo["inconsistent_orientation_edges"]:
            hard.append(f"{topo['inconsistent_orientation_edges']} shell edges with inconsistent normals")
        mx = cfg["checks"]["max_free_edges"]
        if mx >= 0 and topo["free_edges"] > mx:
            hard.append(f"{topo['free_edges']} free edges (allowed {mx})")
    ok = not hard and n_bad == 0
    return Report(ok, hard, n_bad, stats, topo, rows, bad_info, counts)


# Worst-case relative coordinate error each GRID format is expected to stay within.
# "small" (8-char fields) only holds ~6-7 significant digits: use large or free for precision.
FMT_TOL = {"large": 1e-9, "free": 1e-12, "small": 1e-5}


# --------------------------------------------------------------------------------------
# BDF writer
# --------------------------------------------------------------------------------------
def _real(x: float, w: int) -> str:
    """Shortest-loss real number that fits w columns (fixed or exponent form, whichever is closer)."""
    x = float(x)
    if x == 0.0:
        return "0."
    cands = []
    for k in range(w, -1, -1):
        s = f"{x:.{k}f}" + ("." if k == 0 else "")
        if len(s) <= w:
            cands.append(s)
            break
    for p in range(w, -1, -1):
        m, ex = f"{x:.{p}E}".split("E")
        s = f"{m}{'.' if p == 0 else ''}E{int(ex):+d}"
        if len(s) <= w:
            cands.append(s)
            break
    return min(cands, key=lambda s: abs(float(s) - x))


def _free_real(x: float) -> str:
    s = f"{float(x):.15g}"
    return s if any(ch in s for ch in ".eE") else s + "."


def _fixed(name: str, vals: list, w: int, per: int) -> list:
    lines = [(name.ljust(8) + "".join(v.rjust(w) for v in vals[:per])).rstrip()]
    rest = vals[per:]
    while rest:
        lines.append((("*" if w == 16 else "").ljust(8) + "".join(v.rjust(w) for v in rest[:per])).rstrip())
        rest = rest[per:]
    return lines


def _free(name: str, vals: list) -> list:
    lines = [",".join([name] + vals[:8])]
    for i in range(8, len(vals), 8):
        lines.append("," + ",".join(vals[i:i + 8]))
    return lines


def write_bdf(path: Path, mesh: Mesh, cfg: dict, mats: dict, rules: list, banner: list[str]) -> None:
    fmt, deck = cfg["output"]["format"], cfg["output"]["deck"]
    I = lambda v: str(int(v))
    L: list[str] = [f"$ {b}" for b in banner]
    if deck == "full":
        L += ["SOL 101", "CEND", f"TITLE = {cfg['output']['title']}", "BEGIN BULK"]
    # nodes
    for i in range(len(mesh.xyz)):
        x, y, z = mesh.xyz[i]
        if fmt == "large":
            # 15 chars of number in a 16-char field: always one blank column between fields
            L += _fixed("GRID*", [I(mesh.nid[i]), "", _real(x, 15), _real(y, 15), _real(z, 15)], 16, 4)
        elif fmt == "small":
            L += _fixed("GRID", [I(mesh.nid[i]), "", _real(x, 8), _real(y, 8), _real(z, 8)], 8, 8)
        else:
            L += _free("GRID", [I(mesh.nid[i]), "", _free_real(x), _free_real(y), _free_real(z)])
    # elements
    w, per = (8, 8) if fmt != "free" else (0, 0)
    for t in mesh.types():
        for eid, pid, c in zip(mesh.eid[t], mesh.pid[t], mesh.conn[t]):
            vals = [I(eid), I(pid)] + [I(mesh.nid[j]) for j in c]
            L += _fixed(CARD[t], vals, 8, 8) if fmt != "free" else _free(CARD[t], vals)
    # properties + materials actually used
    used = sorted({int(p) for t in mesh.types() for p in np.unique(mesh.pid[t])})
    solid = any(KIND[t] == "tet" for t in mesh.types())
    used_mats = set()
    for pid in used:
        r = rules[pid - 1]
        used_mats.add(r["material"])
        L.append(f"$ PID {pid}: parts matching /{r['match']}/" + ("  ** PLACEHOLDER **" if r.get("placeholder") else ""))
        if solid:
            vals = [I(pid), I(r["material"])]
            L += _fixed("PSOLID", vals, 8, 8) if fmt != "free" else _free("PSOLID", vals)
        else:
            R = (lambda v: _real(v, 8)) if fmt != "free" else _free_real
            m = I(r["material"])
            vals = [I(pid), m, R(r["thickness"]), m, R(1.0), m, R(0.833333)]
            L += _fixed("PSHELL", vals, 8, 8) if fmt != "free" else _free("PSHELL", vals)
    for mid in sorted(used_mats):
        mt = mats[mid]
        R = (lambda v: _real(v, 8)) if fmt != "free" else _free_real
        vals = [I(mid), R(mt["E"]), "", R(mt["nu"]), R(mt["rho"])]
        L.append(f"$ {mt.get('name', 'MAT1')}")
        L += _fixed("MAT1", vals, 8, 8) if fmt != "free" else _free("MAT1", vals)
    if deck == "full":
        L.append("ENDDATA")
    Path(path).write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------------------
# BDF reader (small / large / free field, continuation lines) - used to certify the written file
# --------------------------------------------------------------------------------------
_REAL = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))(?:[eEdD]?([+-]\d+)|[eEdD]([+-]?\d+))?$")


def parse_real(s: str) -> float:
    s = s.strip()
    m = _REAL.match(s)
    if not m:
        raise ValueError(f"bad real '{s}'")
    ex = m.group(2) or m.group(3)
    return float(m.group(1)) * (10.0 ** int(ex) if ex else 1.0)


def _logical_cards(path: Path):
    cards, cur, in_bulk, saw_begin = [], None, False, False
    text = Path(path).read_text(errors="replace").splitlines()
    saw_begin = any(l.strip().upper().startswith("BEGIN BULK") for l in text)
    in_bulk = not saw_begin
    for raw in text:
        line = raw.split("$")[0].rstrip()
        if not line.strip():
            continue
        up = line.strip().upper()
        if not in_bulk:
            in_bulk = up.startswith("BEGIN BULK")
            continue
        if up.startswith("ENDDATA"):
            break
        free = "," in line
        is_cont = line[0] in "+*, \t"
        if is_cont and cur is not None:
            if free:
                f = line.split(",")[1:]
                cur[1].extend([x.strip() for x in f] + [""] * (8 - len(f)) if len(f) < 8 else [x.strip() for x in f])
            else:
                step = 16 if line[0] == "*" else 8
                body = line[8:72].ljust(64)
                cur[1].extend(body[i:i + step].strip() for i in range(0, 64, step))
            continue
        if free:
            f = [x.strip() for x in line.split(",")]
            name, data = f[0].rstrip("*").upper(), f[1:]
            data += [""] * (8 - len(data)) if len(data) < 8 else []
        else:
            name = line[:8].strip().rstrip("*").upper()
            step = 16 if line[:8].strip().endswith("*") else 8
            body = line[8:72].ljust(64)
            data = [body[i:i + step].strip() for i in range(0, 64, step)]
        cur = (name, data)
        cards.append(cur)
    return cards


def read_bdf(path) -> Mesh:
    nodes, elems, skipped = {}, {t: [] for t in ORDER}, {}
    for name, f in _logical_cards(Path(path)):
        try:
            if name == "GRID":
                nodes[int(f[0])] = (parse_real(f[2]), parse_real(f[3]), parse_real(f[4]))
            elif name == "CTRIA3":
                elems["tria3"].append([int(x) for x in f[:5]])
            elif name == "CQUAD4":
                elems["quad4"].append([int(x) for x in f[:6]])
            elif name == "CTRIA6" and all(f[2:8]):
                elems["tria6"].append([int(x) for x in f[:8]])
            elif name == "CQUAD8" and all(f[2:10]):
                elems["quad8"].append([int(x) for x in f[:10]])
            elif name == "CTETRA":
                if len(f) > 7 and f[6] and f[7]:
                    elems["tet10"].append([int(x) for x in f[:12]])
                else:
                    elems["tet4"].append([int(x) for x in f[:6]])
            elif name.startswith("C") and name not in ("CEND",):
                skipped[name] = skipped.get(name, 0) + 1
        except (ValueError, IndexError) as exc:
            raise CadError(f"cannot parse {name} card in {path}: {exc}")
    ids = np.array(sorted(nodes), np.int64)
    if not len(ids):
        raise CadError(f"no GRID cards found in {path}")
    xyz = np.array([nodes[i] for i in ids], float)
    mesh = Mesh(ids, xyz)
    for t, rows in elems.items():
        if not rows:
            continue
        a = np.array(rows, np.int64)
        idx = np.searchsorted(ids, a[:, 2:])
        if (idx >= len(ids)).any() or (ids[np.minimum(idx, len(ids) - 1)] != a[:, 2:]).any():
            raise CadError(f"{CARD[t]} cards reference nodes that don't exist")
        mesh.conn[t], mesh.eid[t], mesh.pid[t] = idx, a[:, 0], a[:, 1]
    mesh.skipped = skipped
    if not mesh.conn:
        raise CadError("no supported elements (CTRIA3/CQUAD4/CTRIA6/CQUAD8/CTETRA) found")
    return mesh


# --------------------------------------------------------------------------------------
# CAD -> mesh (gmsh / OpenCASCADE)
# --------------------------------------------------------------------------------------
class CadMesher:
    CAD_EXT = {".step", ".stp", ".iges", ".igs", ".brep", ".brp"}

    def __init__(self, path: Path, cfg: dict, say=print, gmsh_module=None, verbose=False):
        self.cfg, self.say, self.fields = cfg, say, []
        if gmsh_module is None:
            try:
                import gmsh as gmsh_module
            except ImportError:
                raise CadError("the 'gmsh' Python package is required:  pip install gmsh")
        self.g = g = gmsh_module
        ext = Path(path).suffix.lower()
        if ext not in self.CAD_EXT:
            raise CadError(f"'{ext}' is not read directly. Export STEP (AP203/214/242) from your CAD system "
                           f"(CATIA, NX, SolidWorks, Creo...) - supported: {', '.join(sorted(self.CAD_EXT))}")
        g.initialize()
        g.option.setNumber("General.Terminal", 1 if verbose else 0)
        g.option.setNumber("Geometry.OCCImportLabels", 1)
        if cfg["cad"]["target_unit"]:
            try:
                g.option.setString("Geometry.OCCTargetUnit", cfg["cad"]["target_unit"])
            except Exception:
                pass
        g.model.add("cad")
        shell = cfg["mesh"]["type"] == "shell"
        try:
            g.model.occ.importShapes(str(path), True)
            g.model.occ.synchronize()
            if cfg["cad"]["heal"]:
                g.model.occ.healShapes([], cfg["cad"]["heal_tolerance"], True, True, True, True, not shell)
                g.model.occ.synchronize()
            g.model.occ.removeAllDuplicates()
            g.model.occ.synchronize()
            if cfg["cad"]["conformal"]:
                ents = g.model.getEntities(2 if shell else 3)
                if len(ents) > 1:
                    g.model.occ.fragment(ents, [])
                    g.model.occ.synchronize()
        except Exception as exc:
            raise CadError(f"could not import/heal CAD: {exc}")
        n3, n2 = len(g.model.getEntities(3)), len(g.model.getEntities(2))
        say(f"  CAD: {n3} solids, {n2} faces")
        if shell and n3 and not cfg["mesh"]["use_skin"]:
            raise CadError(
                "mesh.type = 'shell' but this CAD contains SOLIDS. A shell analysis needs midsurface geometry; "
                "meshing a solid's outer faces gives a double-walled skin, not a midsurface. Options: supply "
                "midsurface STEP, use mesh.type = 'tet', or set mesh.use_skin = true if you really want the skin.")
        if not shell and not n3:
            raise CadError("mesh.type = 'tet' but the CAD contains no solids (open surfaces / unsewn shells?). "
                           "Try cad.heal = true with a larger heal_tolerance, or mesh as 'shell'.")
        if not (n2 if shell else n3):
            raise CadError("no usable geometry after import")

    def close(self):
        try:
            self.g.finalize()
        except Exception:
            pass

    # ---- one meshing attempt ---------------------------------------------------------
    def generate(self, attempt: int, factor: float, balls: list, rules: list) -> Mesh:
        g, cfg = self.g, self.cfg
        mc = cfg["mesh"]
        shell, qm = mc["type"] == "shell", mc["quad_mode"]
        target = mc["target_size"] * factor
        try:
            g.model.mesh.clear()
            self._set_fields(balls, target)
            g.option.setNumber("Mesh.MeshSizeMax", target)
            g.option.setNumber("Mesh.MeshSizeMin", min(mc["min_size"], target))
            g.option.setNumber("Mesh.MeshSizeFromCurvature", mc["curvature_elements"] / factor)
            g.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            algo2d = ([8, 6, 5, 1] if qm != "none" and shell else [6, 5, 1, 6])[attempt % 4]
            g.option.setNumber("Mesh.Algorithm", algo2d)
            g.option.setNumber("Mesh.Algorithm3D", [1, 4, 10, 1][attempt % 4])
            g.option.setNumber("Mesh.RecombineAll", 1 if (shell and qm != "none") else 0)
            g.option.setNumber("Mesh.RecombinationAlgorithm", 3 if qm == "all" else 1)
            g.option.setNumber("Mesh.SecondOrderIncomplete", 1)
            g.option.setNumber("Mesh.SecondOrderLinear", 0)
            g.option.setNumber("Mesh.HighOrderOptimize", 1 if attempt < 2 else 2)
            g.model.mesh.generate(2 if shell else 3)
            for method in (["Laplace2D", "Relocate2D"] if shell else ["Netgen", "Relocate3D"]):
                try:
                    g.model.mesh.optimize(method)
                except Exception:
                    pass   # optimiser not built into this gmsh; the quality loop still enforces criteria
            if mc["order"] == 2:
                g.model.mesh.setOrder(2)
            return self._extract(rules)
        except CadError:
            raise
        except Exception as exc:
            raise MeshingFailure(f"gmsh failed on attempt {attempt + 1}: {exc}")

    def _set_fields(self, balls, target):
        f = self.g.model.mesh.field
        ids = []
        for (x, y, z, r, s) in balls:
            b = f.add("Ball")
            for k, v in dict(Radius=r, VIn=s, VOut=target, XCenter=x, YCenter=y, ZCenter=z).items():
                f.setNumber(b, k, v)
            try:
                f.setNumber(b, "Thickness", 2 * r)
            except Exception:
                pass
            ids.append(b)
        if ids:
            m = f.add("Min")
            f.setNumbers(m, "FieldsList", ids)
            f.setAsBackgroundMesh(m)

    def _extract(self, rules) -> Mesh:
        g, cfg = self.g, self.cfg
        shell = cfg["mesh"]["type"] == "shell"
        dim = 2 if shell else 3
        scale = cfg["mesh"]["length_scale"]
        tags, xyz, _ = g.model.mesh.getNodes()
        tags = np.asarray(tags, np.int64)
        xyz = np.asarray(xyz, float).reshape(-1, 3) * scale
        lut = np.full(int(tags.max()) + 1, -1, np.int64)
        lut[tags] = np.arange(len(tags))
        conn = {t: [] for t in ORDER}
        pid = {t: [] for t in ORDER}
        unmatched, ignored = set(), 0
        for d, tag in g.model.getEntities(dim):
            name = g.model.getEntityName(d, tag) or f"{'SURF' if d == 2 else 'VOL'}_{tag}"
            ri = next((i for i, r in enumerate(rules) if re.search(r["match"], name)), None)
            if ri is None:
                unmatched.add(name)
                continue
            types, _, nodes = g.model.mesh.getElements(d, tag)
            for et, nt in zip(types, nodes):
                t = GMSH_TYPE.get(int(et))
                if t is None:
                    ignored += len(nt)
                    continue
                c = lut[np.asarray(nt, np.int64)].reshape(-1, GMSH_NODES.get(int(et), NNODES[t]))[:, :NNODES[t]]
                if t in GMSH_PERM:
                    c = c[:, GMSH_PERM[t]]
                conn[t].append(c)
                pid[t].append(np.full(len(c), ri + 1, np.int64))
        if unmatched:
            raise CadError("no [[property]] rule matches these CAD parts: " + ", ".join(sorted(unmatched)))
        m = Mesh(tags, xyz)
        for t in ORDER:
            if conn[t]:
                m.conn[t] = np.concatenate(conn[t])
                m.pid[t] = np.concatenate(pid[t])
                m.eid[t] = np.zeros(len(m.conn[t]), np.int64)
        return m


def refinement_balls(prev: list, rep: Report, mesh: Mesh, floor: float, cap: int = 300) -> list:
    """Cluster violating elements into spherical size fields (x, y, z, radius, size)."""
    if not rep.bad_info:
        return prev
    cen = np.concatenate([c for c, _ in rep.bad_info])
    size = np.concatenate([s for _, s in rep.bad_info])
    size = np.where(size > 0, size, np.median(size[size > 0]) if (size > 0).any() else floor)
    cell = 2.0 * float(np.median(size))
    while True:
        key = np.floor(cen / cell).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        if len(uniq) <= cap:
            break
        cell *= 2
    inv = inv.reshape(-1)
    new = []
    for k in range(len(uniq)):
        sel = inv == k
        c = cen[sel].mean(0)
        s = max(float(size[sel].min()) * 0.7, floor)
        new.append((float(c[0]), float(c[1]), float(c[2]), cell, s))
    return (prev + new)[-cap:]


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
def fmt_limit(lo, hi):
    return " & ".join(x for x in ([f">= {lo:g}"] if lo is not None else []) + ([f"<= {hi:g}"] if hi is not None else [])) or "-"


def print_report(rep: Report, say=print):
    say("  element counts: " + ", ".join(f"{t}={n}" for t, n in rep.counts.items()))
    for t, ms in rep.stats.items():
        say(f"  {t}")
        say(f"    {'metric':<14}{'min':>11}{'mean':>11}{'max':>11}   {'limit':<16}{'fails':>7}")
        for name, s in ms.items():
            if s["min"] is None:
                continue
            lim = fmt_limit(s["min_limit"], s["max_limit"])
            say(f"    {name:<14}{s['min']:>11.4g}{s['mean']:>11.4g}{s['max']:>11.4g}   {lim:<16}{s['fails']:>7}")
    say("  topology: " + ", ".join(f"{k}={v}" for k, v in rep.topology.items()))
    for h in rep.hard:
        say(f"  FAIL: {h}")


def write_violations(path: Path, rep: Report) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["element_type", "eid", "pid", "metric", "value", "min_limit", "max_limit", "cx", "cy", "cz"])
        w.writerows(rep.rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_mesh(cad: Path, out: Path, cfg: dict, force=False, say=print, gmsh_module=None, verbose=False) -> int:
    t0 = time.time()
    say(f"cad2bdf {__version__}: {cad.name} -> {out.name}  [{cfg['mesh']['type']}, order {cfg['mesh']['order']}, "
        f"target {cfg['mesh']['target_size']:g}]")
    mats = {m["id"]: m for m in cfg["material"]}
    rules = cfg["property"]
    if any(r.get("placeholder") for r in rules):
        say("  WARNING: using the PLACEHOLDER material/property. Define [[material]]/[[property]] in your config.")
    mesher = CadMesher(cad, cfg, say, gmsh_module, verbose)
    try:
        best, balls, history, since_best, last_error = None, [], [], 0, ""
        it_cfg = cfg["iterate"]
        for it in range(it_cfg["max_iterations"]):
            factor = it_cfg["shrink"] ** it
            say(f"  attempt {it + 1}/{it_cfg['max_iterations']}  (size x{factor:.3f}, {len(balls)} local refinement zones)")
            try:
                mesh = cleanup(mesher.generate(it, factor, balls, rules), cfg, say)
            except MeshingFailure as exc:
                say(f"    {exc}")
                last_error = str(exc)
                history.append(dict(attempt=it + 1, error=last_error))
                continue
            rep = evaluate(mesh, cfg)
            say(f"    {mesh.n_elem()} elements, {len(mesh.xyz)} nodes -> "
                f"{rep.n_bad} elements violate criteria, {len(rep.hard)} structural failures")
            history.append(dict(attempt=it + 1, size_factor=factor, elements=mesh.n_elem(),
                                violating_elements=rep.n_bad, structural_failures=rep.hard))
            if best is None or rep.score < best[1].score:
                best, since_best = (mesh, rep), 0
            else:
                since_best += 1
            if rep.ok:
                break
            if since_best >= 3:
                say("    no improvement for 3 attempts - stopping")
                break
            if it_cfg["local_refine"]:
                balls = refinement_balls(balls, rep, mesh, cfg["mesh"]["min_size"])
        if best is None:
            raise CadError(f"meshing failed on every attempt. Last error: {last_error}")
    finally:
        mesher.close()

    mesh, rep = best
    stem = out.with_suffix("")
    certified = rep.ok
    if not certified:
        write_violations(Path(f"{stem}.violations.csv"), rep)
        say("\nCRITERIA NOT MET on the best attempt:")
        print_report(rep, say)
        say(f"  violations listed in {stem}.violations.csv")
        if not force:
            _write_json(Path(f"{stem}.quality.json"), rep, cfg, cad, None, history, False)
            say("No BDF written (use --force to write the best attempt anyway, clearly marked as failing).")
            return 2

    banner = [f"Generated by cad2bdf {__version__} from {cad.name} (sha256 {sha256(cad)[:16]}...)",
              f"Units = CAD units x {cfg['mesh']['length_scale']:g}"]
    banner.append("QUALITY CRITERIA SATISFIED (verified on this file)" if certified
                  else f"*** WARNING: QUALITY CRITERIA NOT MET - {rep.n_bad} violating elements, "
                       f"{len(rep.hard)} structural failures ***")
    write_bdf(out, mesh, cfg, mats, rules, banner)

    # independent certification: re-read the *file* and re-evaluate
    back = read_bdf(out)
    rep2 = evaluate(back, cfg)
    fidelity = None
    if back.n_elem() == mesh.n_elem() and len(back.xyz) == len(mesh.xyz):
        fidelity = float(np.abs(back.xyz - mesh.xyz).max())
    verified = (rep2.ok == rep.ok and rep2.n_bad == rep.n_bad and back.n_elem() == mesh.n_elem()
                and fidelity is not None and fidelity <= FMT_TOL[cfg["output"]["format"]] * max(1.0, float(np.abs(mesh.xyz).max())))
    say("\nResult:")
    print_report(rep2, say)
    say(f"  file round-trip: max coordinate error {fidelity:.3g}" if fidelity is not None else "  file round-trip: MISMATCH")
    if not verified:
        say("  ERROR: the written BDF does not match the in-memory mesh - not trusting it.")
        return 1
    _write_json(Path(f"{stem}.quality.json"), rep2, cfg, cad, out, history, certified)
    say(f"\n{'CERTIFIED' if certified else 'WRITTEN (NOT CERTIFIED)'}: {out}  ({time.time() - t0:.1f}s)")
    return 0 if certified else 2


def _write_json(path: Path, rep: Report, cfg: dict, cad: Path, bdf: Path | None, history: list, certified: bool):
    doc = dict(tool=f"cad2bdf {__version__}", certified=certified, cad=cad.name, cad_sha256=sha256(cad),
               bdf=bdf.name if bdf else None, bdf_sha256=sha256(bdf) if bdf else None,
               counts=rep.counts, structural_failures=rep.hard, violating_elements=rep.n_bad,
               topology=rep.topology, stats=rep.stats, criteria=cfg["quality"], checks=cfg["checks"],
               history=history)
    path.write_text(json.dumps(doc, indent=2))


def run_check(bdf: Path, cfg: dict, say=print) -> int:
    mesh = read_bdf(bdf)
    if getattr(mesh, "skipped", None):
        say("  note: not evaluated (unsupported cards): " + ", ".join(f"{k}x{v}" for k, v in mesh.skipped.items()))
    rep = evaluate(mesh, cfg)
    say(f"{bdf.name}: {len(mesh.xyz)} nodes, {mesh.n_elem()} elements")
    print_report(rep, say)
    stem = bdf.with_suffix("")
    write_violations(Path(f"{stem}.violations.csv"), rep)
    say(f"{'PASS' if rep.ok else 'FAIL'}: {rep.n_bad} violating elements, {len(rep.hard)} structural failures"
        + ("" if rep.ok else f" (details: {stem}.violations.csv)"))
    return 0 if rep.ok else 2


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------
def _grid_plate(nx, ny, quads=True):
    xs, ys = np.meshgrid(np.arange(nx + 1, dtype=float), np.arange(ny + 1, dtype=float))
    xyz = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
    ix = lambda i, j: j * (nx + 1) + i
    q = np.array([[ix(i, j), ix(i + 1, j), ix(i + 1, j + 1), ix(i, j + 1)] for j in range(ny) for i in range(nx)])
    m = Mesh(np.arange(1, len(xyz) + 1), xyz)
    if quads:
        m.conn["quad4"] = q
    else:
        m.conn["tria3"] = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    for t in m.conn:
        m.eid[t], m.pid[t] = np.arange(len(m.conn[t])) + 1, np.ones(len(m.conn[t]), np.int64)
    return renumber(m)


def selftest() -> int:
    fails = []

    def check(name, cond):
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    print("metrics on reference shapes")
    s3 = math.sqrt(3) / 2
    m = tri_metrics(np.array([[[0, 0, 0], [1, 0, 0], [0.5, s3, 0]]], float))
    check("equilateral tria: 60/60 deg, aspect 1.155, skew 0",
          abs(m["min_angle"][0] - 60) < 1e-9 and abs(m["aspect"][0] - 2 / math.sqrt(3)) < 1e-9 and abs(m["skew"][0]) < 1e-6)
    m = quad_metrics(np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]], float))
    check("unit square: 90 deg, aspect 1, skew/taper/warp 0, jacobian 1",
          abs(m["min_angle"][0] - 90) < 1e-9 and abs(m["aspect"][0] - 1) < 1e-12 and abs(m["skew"][0]) < 1e-9
          and abs(m["taper"][0]) < 1e-12 and abs(m["warp"][0]) < 1e-12 and abs(m["jacobian"][0] - 1) < 1e-12)
    m = quad_metrics(np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0.1], [0, 1, 0]]], float))
    check("warped quad: warp ~0.035", 0.034 < m["warp"][0] < 0.037)
    m = quad_metrics(np.array([[[0, 0, 0], [2, 0, 0], [1.5, 1, 0], [0.5, 1, 0]]], float))
    check("2:1 trapezoid: taper 1/3", abs(m["taper"][0] - 1 / 3) < 1e-9)
    m = quad_metrics(np.array([[[0, 0, 0], [1, 0, 0], [0.2, 0.2, 0], [0, 1, 0]]], float))
    check("concave quad: negative jacobian", m["jacobian"][0] < 0)
    reg = np.array([[[1, 1, 1], [1, -1, -1], [-1, -1, 1], [-1, 1, -1]]], float)   # (x2-x1)x(x3-x1).(x4-x1) > 0
    m = tet_metrics(reg)
    check("regular tet: quality 1, dihedral 70.53, aspect 1.225, collapse 1.241",
          abs(m["quality"][0] - 1) < 1e-9 and abs(m["min_dihedral"][0] - 70.5288) < 1e-3
          and abs(m["aspect"][0] - 1.2247) < 1e-3 and abs(m["collapse"][0] - 1.2408) < 1e-3)
    m = tet_metrics(reg[:, [1, 0, 2, 3]])   # swap two nodes -> inverted
    check("inverted tet: negative quality and collapse", m["quality"][0] < 0 and m["collapse"][0] < 0)

    print("evaluation + enforcement")
    cfg = load_config(None)
    plate = _grid_plate(6, 4)
    check("regular quad plate passes defaults", evaluate(plate, cfg).ok)
    bad = _grid_plate(6, 4)
    bad.xyz[bad.xyz[:, 0] == 3][:, 2]
    bad.xyz[10, 2] = 0.9
    r = evaluate(bad, cfg)
    check("raised node flagged (warp) with element ids", (not r.ok) and any(row[3] == "warp" for row in r.rows))
    cfg2 = load_config(None)
    cfg2["quality"]["quad"]["warp"] = {"max": 5.0}
    cfg2["quality"]["quad"]["jacobian"] = {"min": -1.0}
    cfg2["quality"]["quad"]["taper"] = {"max": 5.0}
    cfg2["quality"]["quad"]["min_angle"] = {"min": 1.0}
    cfg2["quality"]["quad"]["max_angle"] = {"max": 179.0}
    cfg2["quality"]["quad"]["skew"] = {"max": 89.0}
    check("loosened criteria accept the same mesh", evaluate(bad, cfg2).ok)
    try:
        bad_cfg = {"quality": {"quad": {"warpp": {"max": 1}}}}
        c = deep_merge(tomllib.loads(DEFAULT_TOML), bad_cfg)
        validate_config(c)
        check("typo in metric name is rejected", False)
    except CadError:
        check("typo in metric name is rejected", True)

    print("topology")
    tp = _grid_plate(5, 5, quads=False)
    tp.conn["tria3"][::2] = tp.conn["tria3"][::2][:, [0, 2, 1]]
    r = evaluate(tp, cfg)
    check("flipped trias detected as orientation failure", any("inconsistent" in h for h in r.hard))
    fix_orientation(tp)
    check("fix_orientation repairs it", evaluate(tp, cfg).topology["inconsistent_orientation_edges"] == 0)

    print("2nd-order ordering")
    X = reg[0]
    pairs = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    xyz = np.vstack([X] + [0.5 * (X[i] + X[j]) for i, j in pairs])
    t10 = Mesh(np.arange(1, 11), xyz, {"tet10": np.arange(10)[None, :]}, {"tet10": np.array([1])}, {"tet10": np.array([1])})
    check("correct Nastran CTETRA10 order accepted", midside_order_errors(t10) == 0)
    t10.conn["tet10"] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 9, 8]])
    check("swapped mid-nodes caught", midside_order_errors(t10) == 1)

    print("BDF write -> read round trip")
    import tempfile
    mats, rules = {1: dict(E=71700.0, nu=0.33, rho=2.81e-9, name="AL")}, [dict(match=".*", material=1, thickness=1.5)]
    rng = np.random.default_rng(1)
    for fmt in ("large", "small", "free"):
        for kind in ("quad", "tria", "tet10"):
            if kind == "tet10":
                mesh = Mesh(np.arange(1, 11), xyz * 123.456789012345 + 1e-3, {"tet10": np.arange(10)[None, :]},
                            {"tet10": np.array([7])}, {"tet10": np.array([1])})
            else:
                mesh = _grid_plate(7, 5, quads=(kind == "quad"))
                mesh.xyz = mesh.xyz * 1234.56789012345 + rng.normal(size=mesh.xyz.shape) * 0.01
            c = load_config(None)
            c["output"]["format"] = fmt
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "m.bdf"
                write_bdf(p, mesh, c, mats, rules, ["selftest"])
                back = read_bdf(p)
                tol = FMT_TOL[fmt] * np.abs(mesh.xyz).max()
                same = (back.n_elem() == mesh.n_elem()
                        and all((back.conn[t] == mesh.conn[t]).all() for t in mesh.conn)
                        and np.abs(back.xyz - mesh.xyz).max() <= tol)
                widths = max(len(l) for l in p.read_text().splitlines()) <= 80
            check(f"{fmt:<5} {kind:<5}: lossless connectivity, coordinates within {tol:.1e}, lines <= 80 cols",
                  same and widths)

    try:
        import gmsh  # noqa: F401
    except ImportError:
        print("gmsh not installed -> skipping CAD end-to-end tests (pip install gmsh, then re-run selftest)")
        return 1 if fails else 0

    print("CAD end-to-end (real gmsh / OpenCASCADE)")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("blk")
        box = gmsh.model.occ.addBox(0, 0, 0, 40, 30, 20)
        cyl = gmsh.model.occ.addCylinder(20, 15, -1, 0, 0, 22, 6)
        gmsh.model.occ.cut([(3, box)], [(3, cyl)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(d / "block.step"))
        gmsh.model.remove()
        gmsh.model.add("plate")
        pl = gmsh.model.occ.addRectangle(0, 0, 0, 100, 60)
        dk = gmsh.model.occ.addDisk(50, 30, 0, 10, 10)
        gmsh.model.occ.cut([(2, pl)], [(2, dk)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(d / "plate.step"))
        gmsh.finalize()
        for name, mt, order, size in (("block.step", "tet", 1, 4.0), ("block.step", "tet", 2, 5.0),
                                      ("plate.step", "shell", 1, 5.0), ("plate.step", "shell", 2, 6.0)):
            c = load_config(None)
            c["mesh"].update(type=mt, order=order, target_size=size, min_size=1.0)
            rc = run_mesh(d / name, d / f"{mt}{order}.bdf", c, say=lambda s: None)
            check(f"{name} as {mt} order {order}: certified", rc == 0)
    return 1 if fails else 0


# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cad2bdf", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mesh", help="CAD file -> certified BDF")
    p.add_argument("cad"); p.add_argument("-c", "--config"); p.add_argument("-o", "--out")
    p.add_argument("--force", action="store_true", help="write the best attempt even if criteria are not met")
    p.add_argument("--verbose", action="store_true", help="show gmsh output")
    p = sub.add_parser("check", help="audit an existing BDF against the criteria")
    p.add_argument("bdf"); p.add_argument("-c", "--config")
    sub.add_parser("init-config", help="print a fully documented template config")
    sub.add_parser("selftest", help="run the built-in verification")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "init-config":
            sys.stdout.write(DEFAULT_TOML)
            return 0
        if a.cmd == "selftest":
            return selftest()
        cfg = load_config(a.config)
        if a.cmd == "check":
            return run_check(Path(a.bdf), cfg)
        cad = Path(a.cad)
        out = Path(a.out) if a.out else cad.with_suffix(".bdf")
        return run_mesh(cad, out, cfg, force=a.force, verbose=a.verbose)
    except CadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
