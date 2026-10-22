"""
Tests for cad2bdf's CAD-facing code path, run against a *stub* of the gmsh Python API so they work
without gmsh installed.  The stub is deliberately imperfect (it meshes a flat plate / a cube), so
these tests prove the glue logic - property mapping, node/element extraction, mid-node permutation,
the retry loop, failure handling, file certification - NOT that gmsh/OpenCASCADE behave as assumed.
That part is covered by `python cad2bdf.py selftest` on a machine with gmsh installed.

Run:  python test_cad2bdf.py      (or: pytest test_cad2bdf.py)
"""
import itertools
import json
import math
import tempfile
from pathlib import Path

import numpy as np

import cad2bdf as c


class _Fields:
    def __init__(self, g): self.g, self.n = g, 0
    def add(self, kind): self.n += 1; self.g.fields_added.append(kind); return self.n
    def setNumber(self, *a): pass
    def setNumbers(self, *a): pass
    def setAsBackgroundMesh(self, *a): self.g.background_set = True
    def remove(self, *a): pass


class FakeGmsh:
    """Minimal stand-in for the gmsh API subset cad2bdf uses.
    mode='plate': 60x40 flat surface; mesh is deliberately bad unless MeshSizeMax < bad_above.
    mode='cube' : 30^3 solid split into Kuhn tets (gmsh node order, incl. tet10 mid-nodes)."""
    __version__ = "stub"

    def __init__(self, mode="plate", bad_above=4.8, name="SKIN_PANEL"):
        self.mode, self.bad_above, self.name = mode, bad_above, name
        self.opts, self.fields_added, self.background_set, self.order, self.generated = {}, [], False, 1, 0
        g = self
        self.option = type("O", (), dict(setNumber=lambda s, k, v: g.opts.__setitem__(k, v),
                                         setString=lambda s, k, v: None))()
        occ = type("OCC", (), dict(importShapes=lambda s, p, h=True: [(2, 1)], healShapes=lambda s, *a: [],
                                   synchronize=lambda s: None, removeAllDuplicates=lambda s: None,
                                   fragment=lambda s, *a: None))()
        mesh = type("M", (), {})()
        mesh.clear = lambda: setattr(g, "_mesh", None)
        mesh.generate = g._generate
        mesh.optimize = lambda *a, **k: None
        mesh.setOrder = g._set_order
        mesh.getNodes = g._get_nodes
        mesh.getElements = g._get_elements
        mesh.field = _Fields(g)
        self.model = type("Mo", (), dict(
            add=lambda s, n: None, occ=occ, mesh=mesh,
            getEntities=lambda s, d: ([(2, 1)] if (d == 2 and g.mode == "plate") else
                                      [(3, 1)] if (d == 3 and g.mode == "cube") else
                                      [(2, 1)] if (d == 2) else []),
            getEntityName=lambda s, d, t: g.name))()

    def initialize(self): pass
    def finalize(self): pass

    # ---- fake meshing --------------------------------------------------------------------
    def _generate(self, dim):
        self.generated += 1
        size = self.opts["Mesh.MeshSizeMax"]
        if self.mode == "plate":
            nx, ny = math.ceil(60 / size), math.ceil(40 / size)
            xs, ys = np.meshgrid(np.linspace(0, 60, nx + 1), np.linspace(0, 40, ny + 1))
            P = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
            if size >= self.bad_above:                       # emulate a bad region at coarse size
                k = (ny // 2) * (nx + 1) + nx // 2
                P[k] += [0.46 * 60 / nx, 0.46 * 40 / ny, 0.0]
            ix = lambda i, j: j * (nx + 1) + i
            q = np.array([[ix(i, j), ix(i + 1, j), ix(i + 1, j + 1), ix(i, j + 1)]
                          for j in range(ny) for i in range(nx)])
            if self.opts.get("Mesh.RecombineAll"):
                self._elems = {3: q}
            else:
                self._elems = {2: np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])}
        else:
            n = max(2, math.ceil(30 / size))
            g = np.linspace(0, 30, n + 1)
            X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
            P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
            ix = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k
            tets = []
            for i, j, k in itertools.product(range(n), repeat=3):
                for perm in itertools.permutations(range(3)):
                    v = [np.array([i, j, k])]
                    for a in perm:
                        w = v[-1].copy(); w[a] += 1; v.append(w)
                    t = [ix(*p) for p in v]
                    a_, b_, c_ = (P[t[m]] - P[t[0]] for m in (1, 2, 3))
                    if np.dot(np.cross(a_, b_), c_) < 0:
                        t[0], t[1] = t[1], t[0]
                    tets.append(t)
            self._elems = {4: np.array(tets)}
        # node tags are NOT contiguous and include an orphan (like gmsh vertex nodes)
        self._P = np.vstack([P, [[-99.0, -99.0, -99.0]]])
        self._tag = np.arange(len(self._P)) * 3 + 5
        self._mesh = True

    def _set_order(self, order):
        self.order = order
        if order != 2 or 4 not in self._elems:
            return
        P, tets, cache, new = list(self._P), self._elems[4], {}, []
        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                cache[k] = len(P); P.append(0.5 * (self._P[a] + self._P[b]))
            return cache[k]
        for t in tets:   # gmsh tet10 ordering: 4:(0,1) 5:(1,2) 6:(2,0) 7:(3,0) 8:(3,2) 9:(3,1)
            new.append(list(t) + [mid(t[0], t[1]), mid(t[1], t[2]), mid(t[2], t[0]),
                                  mid(t[3], t[0]), mid(t[3], t[2]), mid(t[3], t[1])])
        self._P = np.array(P); self._tag = np.arange(len(P)) * 3 + 5
        self._elems = {11: np.array(new)}

    def _get_nodes(self, *a, **k):
        return self._tag, self._P.ravel(), []

    def _get_elements(self, dim, tag):
        types = list(self._elems)
        return types, [np.arange(len(self._elems[t])) + 1 for t in types], \
            [self._tag[self._elems[t]].ravel() for t in types]


def _cfg(**mesh):
    cfg = c.load_config(None)
    cfg["mesh"].update(mesh)
    return cfg


def _run(tmp, cfg, fake, force=False, bdf="out.bdf"):
    cad = Path(tmp) / "part.step"
    cad.write_text("stub")
    logs = []
    rc = c.run_mesh(cad, Path(tmp) / bdf, cfg, force=force, say=logs.append, gmsh_module=fake)
    return rc, logs


def test_retry_loop_reaches_certified_mesh():
    with tempfile.TemporaryDirectory() as d:
        fake = FakeGmsh("plate", bad_above=4.8)
        rc, logs = _run(d, _cfg(type="shell", target_size=5.0, min_size=0.5), fake)
        assert rc == 0, "\n".join(logs)
        rep = json.loads((Path(d) / "out.quality.json").read_text())
        assert rep["certified"] and len(rep["history"]) >= 2          # attempt 1 failed, a later one passed
        assert rep["history"][0]["violating_elements"] > 0
        assert "local refinement" in " ".join(logs) and "Ball" in fake.fields_added and fake.background_set
        assert c.run_check(Path(d) / "out.bdf", c.load_config(None), say=lambda s: None) == 0
        text = (Path(d) / "out.bdf").read_text()
        assert "QUALITY CRITERIA SATISFIED" in text and "GRID*" in text and "CTRIA3" in text


def test_orphan_nodes_dropped_and_ids_contiguous():
    with tempfile.TemporaryDirectory() as d:
        rc, _ = _run(d, _cfg(type="shell", target_size=4.0, min_size=1.0), FakeGmsh("plate"))
        m = c.read_bdf(Path(d) / "out.bdf")
        assert rc == 0 and (m.nid == np.arange(1, len(m.nid) + 1)).all()
        assert not (m.xyz == -99).all(axis=1).any()                  # the orphan node is gone


def test_unmeetable_criteria_refuse_to_certify():
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(type="shell", target_size=4.0, min_size=1.0)
        cfg["quality"]["tria"] = {"aspect": {"max": 1.0}}            # impossible
        cfg["iterate"]["max_iterations"] = 3
        rc, logs = _run(d, cfg, FakeGmsh("plate"))
        assert rc == 2 and not (Path(d) / "out.bdf").exists()
        assert (Path(d) / "out.violations.csv").exists()
        rc, _ = _run(d, cfg, FakeGmsh("plate"), force=True)           # --force writes, loudly marked, still rc 2
        assert rc == 2 and "NOT MET" in (Path(d) / "out.bdf").read_text()
        assert json.loads((Path(d) / "out.quality.json").read_text())["certified"] is False


def test_shell_request_on_solid_geometry_is_refused():
    with tempfile.TemporaryDirectory() as d:
        try:
            _run(d, _cfg(type="shell"), FakeGmsh("cube"))
            assert False, "should have raised"
        except c.CadError as e:
            assert "midsurface" in str(e)


def test_tet10_permutation_and_cards():
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(type="tet", order=2, target_size=10.0, min_size=2.0)
        rc, logs = _run(d, cfg, FakeGmsh("cube"))
        assert rc == 0, "\n".join(logs)
        m = c.read_bdf(Path(d) / "out.bdf")
        assert list(m.conn) == ["tet10"] and m.conn["tet10"].shape[1] == 10
        assert c.midside_order_errors(m) == 0
        txt = (Path(d) / "out.bdf").read_text()
        assert "CTETRA" in txt and "PSOLID" in txt and "MAT1" in txt


def test_unmatched_part_is_an_error_not_a_default():
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(type="shell", target_size=4.0, min_size=1.0)
        cfg["property"] = [dict(match="^WING_", material=1, thickness=2.0)]
        try:
            _run(d, cfg, FakeGmsh("plate", name="FUSELAGE_SKIN"))
            assert False, "should have raised"
        except c.CadError as e:
            assert "FUSELAGE_SKIN" in str(e)


def test_property_rules_and_thickness_written():
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(type="shell", target_size=4.0, min_size=1.0)
        cfg["material"] = [dict(id=7, name="TI64", E=113800.0, nu=0.342, rho=4.43e-9)]
        cfg["property"] = [dict(match="SKIN", material=7, thickness=2.5)]
        rc, _ = _run(d, cfg, FakeGmsh("plate", name="WING_SKIN_L"))
        txt = (Path(d) / "out.bdf").read_text()
        assert rc == 0 and "PSHELL" in txt and "2.500000" in txt and "MAT1           7" in txt and "TI64" in txt


def test_non_step_input_is_rejected_with_guidance():
    try:
        c.CadMesher(Path("x.CATPart"), c.load_config(None), gmsh_module=FakeGmsh())
        assert False
    except c.CadError as e:
        assert "STEP" in str(e)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [ok]   {name}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - bad}/{len(tests)} passed")
    raise SystemExit(1 if bad else 0)
