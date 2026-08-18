# Three test-harness defects that let a green suite mean nothing

Status: found on 2026-08-17 while adding `mur.access: opendap`. **None of these are fixed**; the
MUR work was carried out with `PYTHONPATH=src` as a workaround. They are recorded here in
severity order — Defect A is the one that matters, because it silently invalidates the other two
and every future change besides.

None of them are caused by the MUR change; all three reproduce on a clean tree.

---

## Defect A — `pytest` imports the *installed* package, not `src/`

`coastal_sst_data` is installed **non-editable** into the interpreter's `site-packages`:

```
$ pip show coastal_sst_data
Version: 0.3.1
Location: /Users/johnbuckner/miniconda3/lib/python3.13/site-packages
                                        # ^ no "Editable project location:" line

$ python -c "import coastal_sst_data as c; print(c.__file__)"
/Users/johnbuckner/miniconda3/lib/python3.13/site-packages/coastal_sst_data/__init__.py
```

`[tool.pytest.ini_options]` in `pyproject.toml` sets `testpaths` and `addopts` but **not**
`pythonpath`, and with a `src/` layout there is nothing else to put `src/` ahead of
`site-packages` on `sys.path`. So `tests/` exercise the last-built 0.3.1 wheel and **your working
tree is not under test at all**.

### Why this is worse than it sounds

It does not fail. It passes.

**Current state of this checkout, as of the merge of PR #71 (`mur.access: opendap`):**

```
$ pip show coastal_sst_data | grep Version
Version: 0.3.1                                  # repo is on 0.3.2 (a528dde)

$ python -c "import coastal_sst_data.processes.mur as m; print(hasattr(m, '_ACCESS'))"
False                                           # the merged backend is not there
```

The OPeNDAP backend is committed, merged, and version-bumped — and `import coastal_sst_data`
still hands you 0.3.1 without it. Anyone who pulls `main` and runs the suite is testing the old
wheel.

That is not hypothetical damage. While developing that change I edited
`src/coastal_sst_data/processes/mur.py`, ran the full suite, and got `804 passed, 1 failed` —
the one failure being Defect B. Every one of those 804 tests ran against code that did not
contain the change. And the change had a real bug at that moment: `run()` did
`_ACCESS[ds_cfg["access"]]` while the test helper `_mur_eff` built a `ds` dict with no `access`
key — a guaranteed `KeyError` on the first AoI. Seven tests should have failed. The suite
reported success. The identical command with `PYTHONPATH=src` produced the seven failures
immediately.

Worth stating plainly: a passing suite is currently evidence about the installed wheel, not about
your edits. Anything merged on the strength of a green run since the last `pip install` should be
treated as untested — including PR #71, which was verified only because it happened to be run
under `PYTHONPATH=src`.

### Fix

Either is sufficient; doing both is cheap and makes the suite correct regardless of install state.

```bash
pip install -e .
```

```toml
# pyproject.toml -- belt and braces, and it fixes the checkout for everyone
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
pythonpath = ["src"]      # pytest >= 7.0
```

A guard test is worth considering too — assert that `coastal_sst_data.__file__` resolves inside
the repo, so this can never again be silently true.

---

## Defect B — the golden snapshot fails on a clean tree, over signed zero

```
$ git stash && PYTHONPATH=src python -m pytest \
    tests/test_preprocess.py::test_preprocessed_golden_is_unchanged -q
FAILED — preprocessed cube drifted from the golden snapshot:
  data_vars[tide_coops] VALUES changed:
    stats {'max': 0.0, 'mean': -0.0, 'min': -0.0, 'n_finite': 3}
       -> {'n_finite': 3, 'min': 0.0, 'max': 0.0, 'mean': 0.0}
```

Three finite values, all zero, in `tide_coops`. The committed golden holds `-0.0`; the current
run produces `+0.0`. `-0.0 == 0.0` is `True` — **the data has not changed.**

### Root cause

`tests/test_datacube.py:1293`, `_fingerprint`:

```python
if b.dtype.kind == "f":
    b = b.copy()
    nan = np.isnan(b)
    b[nan] = 0.0          # NaN bit patterns normalized...
    h.update(nan.tobytes())
h.update(b.tobytes())     # ...but -0.0 (0x8000...) still hashes != +0.0 (0x0000...)
```

The function already takes deliberate care that "two arrays agree on which cells are missing and
on every finite value fingerprint identically — regardless of the particular NaN bit pattern a
given code path happened to produce" (its own docstring). Signed zero is exactly the same class
of problem and is not handled: IEEE-754 gives `-0.0` and `+0.0` different bytes and equal value,
so `tobytes()` separates two arrays that compare equal elementwise.

Whichever platform or library version last regenerated the golden produced `-0.0` here; this one
produces `+0.0`. It will keep flipping.

### Fix

One line, alongside the NaN normalization — `b == 0` is `True` for both zeros, so this collapses
them:

```python
b[b == 0] = 0.0           # -0.0 and +0.0 are the same VALUE; hash them the same
```

Then regenerate with `UPDATE_GOLDEN=1` and review the diff. Both goldens
(`datacube_golden.json`, `preprocessed_golden.json`) share this comparator.

**Do not "fix" this by regenerating alone.** That re-pins the current platform's zero signs and
the test breaks again on the next machine. Fix the comparator first.

Until it is fixed, the golden gate — the stated safety net for the assembler refactor
(`docs/refactor-plan-assembler-and-sources.md`) — is red by default, which trains everyone to
ignore it.

---

## Defect C — the test suite writes into a tracked file

Running the suite dirties the working tree:

```
$ git checkout -- path/to && PYTHONPATH=src python -m pytest tests/test_tides.py -q
$ git status --short path/to
 M path/to/data/TIDE/eo_tides/aligned/a1/a1_tides.nc
```

### Root cause

`examples/config.test.yaml:4` sets a **relative** output dir:

```yaml
output_dir: "path/to/data"
```

Tests that load the example config and actually run a stage resolve that against the CWD — the
repo root — so the suite writes real output into the checkout. The resulting file was then
committed (`c01689e`, "updated tides and CMEMS data to be a per source input"), so it is tracked
and every subsequent run shows up as an unrelated modification in `git status`.

`git ls-files "path/to"` returns exactly one entry, so the blast radius is small today. It will
grow the moment another test writes a stage's output through the example config.

### Fix

1. `git rm --cached path/to/data/TIDE/eo_tides/aligned/a1/a1_tides.nc` and add `path/to/` to
   `.gitignore` — generated output should never have been tracked.
2. Better: stop tests writing there at all. Tests that *run* a stage should override
   `output_dir` to `tmp_path`; the example config is a fixture for parsing and `_build_eff`
   assertions, not an output destination. Note `tests/test_mur.py::test_build_eff_maps_example_config`
   already asserts on the literal `Path("path/to/data") / "MUR" / "aligned"`, so it is the
   *running* tests, not the parsing ones, that need the override.
3. Consider making a relative `output_dir` an error, or resolving it against the config file's
   own directory rather than the CWD — a config whose meaning depends on where you invoke it
   from is a footgun beyond the test suite.

---

## Suggested order

A first, and on its own — it changes what every other test result means, including the two below.
Re-run the full suite under a correct install before touching B or C, because the current
`812 passed, 1 failed` baseline was measured with `PYTHONPATH=src` and may not be what a properly
installed checkout produces.
