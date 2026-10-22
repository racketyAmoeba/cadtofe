# cad2bdf — CAD → Nastran BDF with enforced mesh quality

```
pip install gmsh numpy            # scipy optional (faster node merge). Python >= 3.11
python cad2bdf.py selftest        # verifies your install, incl. real CAD end-to-end if gmsh is present
python cad2bdf.py init-config > my.toml     # edit: element size, criteria, materials, properties
python cad2bdf.py mesh part.step -c my.toml -o part.bdf
python cad2bdf.py check other.bdf -c my.toml   # audit ANY existing BDF against the same criteria
```
Exit codes: `0` certified · `1` error · `2` criteria not met.

## What "certified" means
1. Mesh with gmsh/OpenCASCADE, clean up (merge coincident nodes, consistent shell normals).
2. Evaluate **every element** against **your** criteria (`[quality.*]` in the config).
3. Not met → retry with a different algorithm, smaller global size, and spherical refinement zones
   around the violating elements (up to `iterate.max_iterations`).
4. Write the BDF, then **re-read the written file with an independent parser and re-evaluate it**.
   Only if that passes is it called certified. The JSON report records SHA-256 of the CAD and the BDF.
5. If the criteria can't be met you get no BDF (or, with `--force`, a BDF whose header says it failed),
   plus `<name>.violations.csv` listing element id, metric, value, limit and x/y/z of each offender.

A criterion you don't list is not enforced; a misspelt metric name is an error. Defining a
`[quality.tria|quad|tet]` table *replaces* the default for that element kind.

## Output
`CTRIA3 CQUAD4 CTRIA6 CQUAD8 CTETRA(4/10)`, `GRID` (large field by default, full precision),
`PSHELL/PSOLID`, `MAT1`. `deck = "bulk"` gives an INCLUDE-able bulk-only file. Loads/BCs/contact
are yours to add. Note `format = "small"` (8-column fields) only holds ~6–7 significant digits.

## Metric definitions (as implemented — compare with your standard)
| kind | metric | definition |
|---|---|---|
| tria | aspect | longest edge / shortest altitude (equilateral = 1.155) |
| tria | skew | 90° − smallest acute angle between a median and the edge it meets (equilateral = 0) |
| quad | aspect | longer / shorter bimedian |
| quad | skew | 90° − angle between the bimedians |
| quad | taper | max \|corner-triangle area − mean\| / mean (2:1 trapezoid = 0.333) |
| quad | warp | distance between the two diagonals / mean diagonal length |
| quad | jacobian | min / max signed corner-triangle area (negative = concave/inverted) |
| tet | aspect | longest edge / shortest altitude (regular = 1.225) |
| tet | collapse | min over vertices of altitude / √(opposite face area), signed (regular = 1.241) |
| tet | quality | 6√2·V / L_rms³ (regular = 1, inverted < 0) |
| tet | min/max_dihedral | from outward face normals (regular = 70.53°) |
| all 2nd-order | midnode_dev | mid-node offset from edge midpoint / edge length |

Definitions of skew/taper/warp/aspect **differ between pre-processors and solvers**. If your
standard defines one differently, each metric is a few lines in `tri_metrics/quad_metrics/tet_metrics`.
The default thresholds in the config are generic placeholders, not any company's standard.

## Limits you should know about
* **Shell meshing needs midsurface geometry.** gmsh does not extract midsurfaces from solids. A solid
  input with `type = "shell"` is refused (override: `use_skin = true` meshes the outer faces — a
  double-walled skin, not a midsurface). Use a midsurface STEP from your CAD/pre-processor, or mesh solids as `tet`.
* Input is STEP/IGES/BREP. Native CATIA/NX/SolidWorks/Creo files: export STEP first.
* Not generated: hex/penta elements, beams, connectors (RBE/CBUSH), contact, composite layups.
* "Perfect for your criteria" is achieved by *checking and iterating*, not by magic: geometry with
  features smaller than `min_size`, sliver faces or bad topology can make criteria unattainable.
  The tool then tells you where and why instead of writing a bad file. Fix the geometry (defeature,
  stitch) or adjust `min_size`/`target_size`.
* Shell normals are made *consistent*, not outward-facing. Set orientation yourself if pressure loads
  or ply direction depend on it.

## What has and has not been verified
Verified by the included tests (`python cad2bdf.py selftest`, `python test_cad2bdf.py`): all metrics on
reference shapes, enforcement and failure reporting, orientation repair, BDF round-trip in small/large/free
format, parsing of hand-written decks, mid-node ordering checks, and the CAD-facing pipeline against a
**stub** of gmsh's API.
**Not verified:** real gmsh/OpenCASCADE behaviour (not installable where this was written) and acceptance
by an actual Nastran run. First thing to do: run `selftest` with gmsh installed, then load a produced BDF
in your pre-processor or a SOL 101 smoke test. The gmsh→Nastran CTETRA10 node permutation and the
tet positive-volume convention are checked at run time (`midside_order_errors`, inverted-tet metrics),
so a mismatch fails certification instead of passing silently.
