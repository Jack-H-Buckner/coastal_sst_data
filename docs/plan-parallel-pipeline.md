# Plan: Parallelizing the pipeline

## Context

Every stage of the pipeline runs serially today. `pipeline.run_pipeline` is a `for product in
ordered:` loop (`pipeline.py:234`), every module's `run()` is a `for aoi: for granule:` loop, and
both terminal stages are `for name in names:` loops (`processes/datacube.py:1926`,
`processes/preprocess.py:1055`). An exhaustive search for `ThreadPoolExecutor` /
`multiprocessing` / `asyncio` / `dask.distributed` across `src/` returns exactly one hit —
`threading.RLock` in `auth.py:109`, which is defensive against *earthaccess's own* internal thread
pool, not a pool this package creates.

The run is dominated by three things, all confirmed slow: acquisition (network-bound), datacube
assembly, and preprocessing. **Acquisition is the priority**, because assembly is already
memory-limited and has less headroom.

The good news: **the dependency structure needed to parallelize safely already exists and is
declarative.** `ProductSpec.depends_on` (`products.py:200`) gives an explicit DAG, stably
topologically sorted by `pipeline.process_order()` (`pipeline.py:75`). Every module honours one
uniform contract:

```python
rep = module.acquire(project, grids=grids, aois=module_aois, dry_run=..., overwrite=...)
```

Because `acquire()` already takes an **AoI list**, splitting one call over N AoIs into N calls over
one AoI each requires **zero changes inside any process module**. That is the entire lever this plan
pulls.

**Outcome:** an orchestrator that runs independent `(product, AoI)` work concurrently under
per-backend concurrency budgets, with `--jobs 1` reproducing today's exact serial behaviour.

### Decisions locked with the user

- **Orchestrator-level only.** Products and AoIs run concurrently; no process module's `acquire()`
  signature or body changes, and the per-granule loops inside modules stay serial.
- **Threads, not processes** — see §2. This is what keeps one account's API footprint to one
  account's worth.
- **Target is a 200 GB single server**, with a possible AWS migration later. Concurrency limits are
  therefore config, not constants.
- **Step zero is hardening `met.py`** (§0). It is the one remote read in the tree that `net.py` never
  reached, it is already losing data silently today, and concurrency turns its worst failure mode
  from "slow" into "the worker pool shrinks permanently".

---

## 0. Step zero: bring `processes/met.py` under the network hardening

`net.py` was written for exactly this ("every remote read gets a deadline, and every remote read gets
a small number of retries"), and every other backend was brought under it — `earthaccess`, the STAC
searches, `copernicusmarine`, every `/vsicurl` COG read, GMRT, CUDEM. **`met.py` was missed.** It is
the only module that touches the network without a timeout, a retry, or a backoff, and it is used by
*two* products (`met` and `met_overpass`, which calls `met._fetch_one` at
`processes/met_overpass.py:89`).

### What is missing

| # | Site | Gap |
|---|---|---|
| 0a | `_hrrr_fetch_cycle`, `processes/met.py:127-134` — `Herbie(...)` then `H.xarray(search, remove_grib=True)` | No timeout, no retry, no backoff. Herbie's constructor itself does network source-discovery across the AWS/Google/NOMADS mirrors, then `.xarray()` downloads the GRIB subset. |
| 0b | `_era5_store`, `processes/met.py:187-192` — `xr.open_zarr(uri, storage_options={"token": "anon"})` | No retry on the store open; `_ERA5_CACHE` is an unsynchronized module-global holding a live gcsfs/dask handle. |
| 0c | `_fetch_era5`, `processes/met.py:233` — `.load()` | This is where the actual GCS read happens (everything before it is lazy), so a transient 503 lands here unprotected. |
| 0d | **`_fetch_one`, `processes/met.py:261-265`** | `except Exception: log.warning(...); return None` — a blanket swallow. |

### 0d is a correctness bug, not just a robustness gap

The caller tallies that `None` as `rep.fail(f"{name} {src} ref {dstr}", f"{src}: no data")`
(`processes/met.py:379`, and `:413` for the daily mean). So a **lost download and a genuine coverage
gap are recorded with the same words** — "no data" — and the exception text that says which is gone
by then. That is precisely what `net.py`'s docstring exists to prevent:

> A single transient 503 permanently loses that scene/tile. The caller logs a warning and moves on,
> and — because the skip guard treats a written output as done — the day is never retried on a later
> run either. One blip becomes a permanent hole in the cube.

`processes/cmems.py:144-150` already solves this correctly and is the pattern to copy: it separates
"the credential/network failed" from "this model has no cell here", loudly.

### The fix

1. **Wrap the HRRR cycle in `net.retry`**, with the `Herbie(...)` construction *inside* the closure,
   not outside it — same reasoning as `processes/mur.py:248-259`, where the open is inside the retry
   because a handle bound to a dead session cannot be healed afterwards:
   ```python
   got = net.retry(lambda: _hrrr_fetch_cycle(model, cyc, fxx, product, var_keys),
                   what=f"HRRR {model} {cyc:%Y-%m-%dT%H}")
   ```
2. **Wrap `_era5_store`'s `open_zarr` and `_fetch_era5`'s `.load()` in `net.retry`**, and guard
   `_ERA5_CACHE` with a `threading.Lock` — which also closes hazard 3 in §5.
3. **Give both a deadline.** `net.setup_gdal_env()` does not help here: GRIB arrives over
   `requests` and ARCO over gcsfs/aiohttp, neither of which reads the `GDAL_HTTP_*` knobs. Pass
   `timeout=net.HTTP_TIMEOUT_S` through gcsfs `storage_options`.

   > **Correction, found while implementing.** The plan originally proposed a process-wide
   > `socket.setdefaulttimeout()` for the Herbie side. **That does not work** — urllib3 passes its
   > own timeout object to `socket.create_connection` rather than the sentinel that would let the
   > global default apply, so setting one changes nothing. Measured against a blackholed address:
   > `socket.setdefaulttimeout(3)` still took **75 s** to fail. What does work is installing the
   > default where `requests` reads it — a one-time patch of `requests.Session.request` that fills
   > in `timeout` only when the caller passed none (`net.setup_requests_timeout`). Verified: 3 s
   > with the patch, and an explicit `timeout=1` still honoured, so `insitu_ioos`, `tides` and
   > `datum` — which all pass their own — are untouched. It is installed once at `acquire()` time
   > rather than wrapped around a call, because an install/uninstall pair would itself be a race
   > once stages share a pool.
4. **Make `_fetch_one` distinguish its three outcomes**, so the run report stops lying:
   - `None` — genuinely no data here (HRRR off-continent, no ERA5 cell). Today's `"no data"` tally is
     correct for this case and stays.
   - **raise** — the read failed after retries. The caller at `processes/met.py:367`/`:388` and
     `processes/met_overpass.py:89` catches it and records the real reason via `rep.fail`.
   - success — unchanged.

### Test impact

`tests/test_met.py:216` (`test_fetch_one_swallows_a_source_error`) currently **pins the swallow
behaviour** and must be rewritten to assert the new split: a source error propagates and is tallied
with its own reason, while a `None` return still reports "no data". `tests/test_net.py` already has
the fake-clock retry harness to reuse.

### Why this is step zero rather than a nice-to-have

It stands on its own — it removes a silent data-loss class that exists today at `--jobs 1`. But it is
also a hard prerequisite for §2: **a stalled connection with no timeout holds a worker thread
forever, and `ThreadPoolExecutor` cannot kill a running future** (§5, hazard 12). One stall would
permanently shrink the pool; a handful would deadlock the run. Hardening this is what lets the
`herbie` gate rise above 2 later.

---

## 1. The three parallel axes, and what each is worth

| Axis | Unit | Safe? | Value |
|---|---|---|---|
| **Product** | the `depends_on` DAG | Yes — already declared | 8 of 11 products have no dependencies at all |
| **AoI** | `acquire(aois=[one])` | Yes — AoI names are unique project-wide (`config.py:643`) and every output tree is per-AoI | Scales with AoI count |
| **Time** | narrowed `project.time` | Only for per-day/per-scene products — see §6 | The only axis that helps a **single-AoI, multi-year** run |

### The product DAG has only three edges

- `modis` → `landsat` (coincidence filter reads Landsat's aligned files) — `products.py:393`
- `mur` → `{ecostress, landsat, modis}` (`overpass_sensors` reads the sensor dirs) — `products.py:298`
- `met_overpass` → `{ecostress, landsat, modis}` (snapshots taken at overpass instants) — `products.py:446`

Everything else — `bathymetry`, `cmems`, `ecostress`, `landsat`, `met`, `tides`, `landcover`,
`insitu` — is independent and can start immediately.

### Critically: those edges are *per-AoI*, not global

`overpass.days_with_scenes(dirs, name)` and MODIS's `match_landsat` both filter by AoI name. So
`mur(Hobart)` depends on `landsat(Hobart)` — **not** on `landsat(Tamar)`. Modelling the graph over
`(product, aoi)` nodes rather than running global product waves removes every barrier: AoI A can be
downloading MUR while AoI B is still on Landsat.

---

## 2. Threads, not processes — and why that answers the API question

> *"I am not sure if I can send multiple calls to load data from the same API using one account."*

The answer differs by backend (§3), but the **execution-model** decision falls out of it
immediately.

`auth.py` already contains machinery that makes concurrent API use *safe on one account* — and every
piece of it is **process-global, guarded by a thread lock**:

- `_SESSIONS: dict[str, _Session]` (`auth.py:104`) — one credential record per backend
- `_LOCK = threading.RLock()` (`auth.py:109`) — makes the refresh check-and-act atomic
- `MIN_REFRESH_INTERVAL_S = 60.0` (`auth.py:66`) — the login-storm guard: hundreds of simultaneous
  auth failures coalesce onto **one** re-login
- `MAX_REFRESHES = 20` (`auth.py:70`) — per backend, **per process**

The comment at `auth.py:106` says it outright:

> `earthaccess.open()` fans out over a thread pool, so several workers can hit the same dead
> credential at the same instant and all ask to refresh it. The lock makes the rate-limit
> check-and-act atomic, so the first wins and the rest coalesce onto its result.

**Under threads**, N workers share one login, one refresh budget, one rate-limit window — exactly one
account footprint, regardless of worker count. **Under processes**, each worker gets its own
`_SESSIONS`, its own 20-refresh budget and its own 60-second window, so 8 workers become 8
independent login storms against one account. That is precisely the failure mode to avoid.

The GIL is not a real cost here: downloads are I/O-bound, and the heavy CPU work (GDAL/rasterio
reproject, pyresample, numpy, h5netcdf/blosc decompress) releases the GIL.

**Decision: one process, `ThreadPoolExecutor`.** Processes and Slurm array jobs stay available as an
*external* option via the existing `--aoi` flag, which already shards cleanly.

---

## 3. Per-backend concurrency budgets

One global `--jobs` number is wrong, because the backends tolerate wildly different concurrency. Each
task acquires **two** semaphores in a fixed order (backend, then product) so there is no deadlock.

| Backend | Products | Default | Why |
|---|---|---|---|
| `earthdata` | mur, modis, ecostress | **6** | No documented per-account concurrency cap for granule reads; `earthaccess.open()`/`download()` already thread internally, so effective connections are ~6 × its pool. CMR *search* is the throttled part — keep search calls modest. |
| `pc` (Planetary Computer) | landsat, landcover | **4** | Anonymous — **there is no account**, so no per-account limit. Throttling is per-IP, and the STAC search endpoint is the sensitive one. Blob reads scale well. |
| `copernicus` | cmems | **1** | The lazy handle from `copernicusmarine.open_dataset` carries its own client and is not safe to share (`processes/cmems.py:296-319`). The toolbox parallelizes internally already. |
| `noaa_small` | tides, (datum) | **1** | Shared module-global `requests.Session` (`processes/tides.py:128`, `processes/datum.py:113`); small metadata APIs. Cheap anyway — no loss. |
| `erddap` | insitu | **1** → 2-4 later | Shared `Session` at `processes/insitu_ioos.py:68`. Raise once it is thread-local. |
| `herbie` | met, met_overpass | **2** → 4 after §0 | Un-hardened until §0 lands: a stalled connection hangs a worker forever and concurrency multiplies that. Raise once the retry and deadline are in. |
| `dem` | bathymetry | **1** → higher later | Shared CUDEM index cache has a TOCTOU (§5, hazard 6). |

Global default `jobs: 8`. All values live in config and are overridable.

**On the AWS migration:** running in `us-west-2` puts you in-region for Earthdata Cloud's direct S3
access, where throughput is far higher and the `earthdata` budget can go well above 6. These numbers
are config, not constants, precisely so that move is a config edit.

---

## 4. Design

### 4a. New module: `src/coastal_sst_data/scheduler.py`

Small and self-contained. Nothing else in the tree gains a concurrency concept.

```python
@dataclass(frozen=True)
class Task:
    key: tuple            # ("acquire", product, aoi) | ("assemble", aoi) | ("preprocess", aoi)
    deps: tuple           # other task keys
    gates: tuple[str,...] # semaphore names, acquired in sorted order
    run: Callable[[], object]

def run_graph(tasks, *, jobs: int, gates: dict[str, int]) -> dict[key, result | Exception]
```

A Kahn-style loop over a `ThreadPoolExecutor`: submit every task whose deps are complete, and as each
future resolves, submit whatever it unblocked. A task whose dependency **failed** is skipped and
recorded — never silently dropped, matching the existing posture at `pipeline.py:266` that an absent
product must not look like a product that found no data.

### 4b. `pipeline.run_pipeline` builds the graph instead of looping

Reuse, don't replace:

- `compute_grids(project)` — unchanged, still runs **once** on the main thread (`pipeline.py:156`)
- `auth.verify(project, products=ordered)` — unchanged preflight, before any task starts (`pipeline.py:221`)
- `_modules_for(project, product, run_aois)` — unchanged; its `(module, [aoi...])` groups are simply
  **exploded into one task per AoI** (`pipeline.py:137`)
- `products.spec(p).depends_on` — the acquire edges, instantiated per AoI
- `config.required_backend(product, opts)` — picks the backend gate

Task graph per AoI:

```
acquire(p, aoi) for each selected product p, edges from spec.depends_on within the same aoi
        │
        └──▶ assemble(aoi)   [deps: every acquire(*, aoi)]
                  │
                  └──▶ preprocess(aoi)   [dep: assemble(aoi)]
```

Chaining assemble→preprocess *per AoI* is a free win: AoI A preprocesses while AoI B assembles.

**`--jobs 1` takes today's code path verbatim** — the existing product-major loop with one batched
`acquire(aois=[all])` call per module. That keeps a true escape hatch and keeps the existing
`tests/test_pipeline.py` dispatch-order assertions honest.

### 4c. Memory: divide the budget, do not detect it N times

This is the one place parallelism can turn a working run into an OOM, and there is prior art in this
very folder — `plan-bounded-memory-assembly.md` opens with a kernel OOM kill.

`datacube.budget_bytes` (`processes/datacube.py:1543`) detects a budget and **halves it**; on the
200 GB server that is a 100 GB allowance. Four AoIs assembling concurrently would each claim 100 GB.
Divide instead:

- Add an additive keyword to `datacube.assemble(...)` and `preprocess.preprocess(...)`:
  `memory_budget_gb: float | None = None`, threaded into `eff` in each `_build_eff`.
- The orchestrator passes `resolved_budget_gb / assemble_jobs`.
- Default `assemble_jobs: 2` — conservative, because assembly is the memory-limited stage.
- Log the division on one line so a future OOM is diagnosable from the log alone, matching the
  existing budget log at `processes/datacube.py:1950`.

`resolve_block_days` (`processes/datacube.py:1557`) then does the right thing automatically: a
smaller budget yields smaller blocks, still chunk-aligned.

### 4d. Config surface

One new block, following the existing `datacube` / `preprocess` pattern (pydantic, `extra="forbid"`):

```yaml
runtime:
  jobs: 8                 # global worker cap; 1 == today's serial path
  assemble_jobs: 2        # AoIs assembled/preprocessed at once; divides the memory budget
  gates:                  # per-backend caps; omitted keys use the defaults in §3
    earthdata: 6
    pc: 4
    copernicus: 1
```

Plus `--jobs` / `--assemble-jobs` flags on `coastal-sst-data run` (`cli.py:218`) and on the
`pipeline.py` argparse.

---

## 5. Hazards — where this breaks if done naively

Each was found in the code and has a concrete fix. **Items 1-5 are prerequisites.**

| # | Hazard | Location | Fix |
|---|---|---|---|
| 1 | **N logins per backend.** Every `run()` calls `auth.login(...)` unconditionally, so exploding one `acquire()` into 8 per-AoI calls means 8 real logins. | `processes/mur.py:163`, `processes/modis.py:215`, `processes/ecostress.py:109`, `processes/cmems.py:250` | Make `auth.login()` a no-op when `_SESSIONS[backend]` is younger than `MAX_AGE_S`. **One contained change in `auth.py`; no process module is touched.** `verify()` passes `force=True` — a preflight that reused a session would answer a question nobody asked. **The lock must be held across the LOGIN, not just the bookkeeping** — see the correction below. |
| 2 | **Shared `requests.Session` module globals** — not thread-safe (connection pool + cookie jar). | `processes/insitu_ioos.py:68`, `processes/tides.py:128`, `processes/datum.py:113` | Gate those products at 1 (§3). Later: `threading.local()` session. |
| 3 | **`_ERA5_CACHE`** — unsynchronized module-global dict holding a live gcsfs/dask handle. | `processes/met.py:184` | Closed by **§0 step 2** (a `threading.Lock` around the cache). |
| 4 | **Lazy module import races.** `_resolve()` calls `importlib.import_module` at dispatch; `tides.py` monkey-patches `collections`/`numpy` at import time (`processes/tides.py:65`). | `pipeline.py:66` | Resolve **every** module on the main thread while building the graph, before the pool starts. Free — `_modules_for` already does the resolution. |
| 5 | **`net.setup_gdal_env()`** mutates `os.environ` from inside several `acquire()`s. | `net.py:102` | Call it once in `run_pipeline` before dispatch. `setdefault` makes it idempotent, so this is tidiness plus one removed race. |
| 6 | **CUDEM index cache TOCTOU** — `exists()` → `unlink()` → `write_text()` → `read_text()`; a second worker can unlink between another's check and read. | `processes/bathymetry_cudem.py:123-139` | Covered by the `dem: 1` gate. Later: a module-level lock around `_fetch_index`, or a per-worker `cudem_index_cache`. |
| 7 | **`store.sweep_scratch` deletes *anyone's* in-flight `.part-*` for the same dest.** | `store.py:93` | ~~Invariant to preserve: no two tasks may target the same output path; add an assertion when building the graph.~~ **That fix was wrong, and the hazard reached production.** The invariant already holds in-process and says nothing about a SECOND PROCESS, which is the case that bit: two runs over one tree (a Slurm array with overlapping `--aoi` lists, two shells, a relaunch) both target the same aligned file, and the sweep unlinked one's scratch out from under an open HDF5 handle — `errno = 2` from `H5FD__sec2_open`, mid-write. Fixed properly by making scratch ATTRIBUTABLE: the tag carries `<host>-<pid>`, and the sweep only deletes what it can show is dead — an in-process registry for our own open writes, `os.kill(pid, 0)` for another pid on this host, and `STALE_SCRATCH_S` against the newest mtime underneath for anything else. See the `WHOSE SCRATCH IS IT` section of the `store` docstring. |
| 7b | **Scratch that is not an `atomic()` destination has the same defect.** A per-granule download dir named `g_<aoi>_<time>` is the same name in every concurrent run, and each run's `finally` rmtrees it. | `processes/modis.py:409`, `processes/modis_ref.py:175` | Put `store.unique_suffix()` in the name. Any scratch path built only from what the ITEM is, rather than from who is processing it, is this bug. |
| 8 | **Scratch tag is `pid`+ms** — no thread component. | `store.py:73` | Add `threading.get_ident()`. Now `<host>-<pid>-<tid>-<ms>`, parsed from the RIGHT so a hyphenated hostname survives — see hazard 7, where the host is what makes "provably dead" provable. |
| 9 | **`_LogOnce` filters** added/removed on shared module loggers; two concurrent AoIs cross-suppress, and one's `removeFilter` un-suppresses the other. | `processes/datacube.py:1705`, `processes/preprocess.py:966` | Refcount the install/remove, or key `seen` by thread. Cosmetic, but it makes logs lie. |
| 10 | **Interleaved logs become unreadable.** No file handler exists — everything is one stderr stream (`cli.py:306`). 8 workers interleaving `[%d/%d] wrote ...` lines is a genuine usability regression. | all entry points | Add a `logging.Filter` that stamps a thread-local `product/aoi` label into the format. The single highest-value ergonomic fix in this plan. |
| 11 | **`ProductReport.elapsed_s` sums on merge** — under parallelism the report's `time` column exceeds wall clock and reads as nonsense. | `report.py:116` | Record `(started_at, ended_at)` and render the span; keep the sum as a separate "busy" figure if useful. |
| 12 | **Ctrl-C no longer cancels promptly.** Today `except KeyboardInterrupt: raise` unwinds cleanly (`pipeline.py:254`); a `ThreadPoolExecutor` cannot kill running futures. | scheduler | `shutdown(wait=False, cancel_futures=True)` on the queued ones, and document that in-flight downloads finish first. `store.atomic` already guarantees no half-written output on interrupt (`store.py:145`). |
| 13 | **`MAX_REFRESHES = 20`** per backend per process may be tight over a long parallel run. | `auth.py:70` | Already config-exposed as `auth.max_refreshes` (`config.py:539`) — raise it for parallel runs and say so in the docs. |
| 14 | **`planetary_computer.sas.TOKEN_CACHE.clear()`** evicts *every* thread's token, not just the failing one. | `auth.py:246` | Accept: `MIN_REFRESH_INTERVAL_S` already caps this at one eviction per minute and the re-sign is cheap. Document it. |

> **Correction to hazard 1, caught by running it.** A freshness check that reads the session and
> *then* authenticates is still a storm on a cold start: every worker reads "no session", every
> worker decides to log in. The first parallel dry run printed `Authenticating with earthdata`
> **twice** — with the fix supposedly in place. `auth._LOCK` has to be held across the
> `authenticate()` call itself, so the first worker logs in while the rest wait and then find a
> credential seconds old. Verified end-to-end: 8 workers × 3 Earthdata products × 2 AoIs now
> produce **exactly one** login, the same as `--jobs 1`.

### Explicitly out of scope — do not parallelize these

- **The blocked Zarr append loop** (`processes/datacube.py:1784`, `processes/preprocess.py:974`).
  Block 0 writes with `mode="w-"` and settles the encoding for the whole cube; later blocks
  `append_zarr(mode="a-")` in order. Parallelizing this means converting to `region=` writes — a real
  refactor with a real corruption risk, and it buys nothing per-AoI parallelism doesn't already give.
- **`assemble(aoi)` and `preprocess(aoi)` for the same AoI.** They read and write the *same*
  `<aoi>.zarr` (`processes/preprocess.py:1057`). The graph edge enforces this.
- **Per-granule loops inside modules** — per the chosen scope.

---

## 6. Optional Phase 3: the temporal axis

Worth building **only if a single-AoI multi-year run is a real workload**, since with several AoIs
the first two axes already saturate 8 workers.

It is still orchestrator-level: narrow `project.time` with `model_copy` and issue several
`acquire()` calls per (product, AoI) over disjoint date shards. No module changes.

**It is safe only for products that write one file per day or per scene**, which is derivable from
`ProductSpec.kind` (`products.py:79`):

- ✅ `DAILY_RASTER`, `OVERPASS_SENSOR`, `OVERPASS_ALIGNED` — mur, cmems, met, ecostress, landsat,
  modis, met_overpass. Disjoint per-day destinations; the skip guard is per file.
- ❌ `SERIES_1D` (tides), `STATION_TABLE` (insitu) — these write **one** file spanning the whole
  range, gated by `requested_start`/`requested_end` attrs via `store._covers_range` (`store.py:250`).
  Shards would overwrite each other.
- ❌ `STATIC_RASTER` (bathymetry, landcover) — no time dimension.

Cost: each shard repeats the catalogue search, so more API calls against a narrower window.

Assembly cannot use this axis — it is already time-blocked, and the blocks are order-dependent
appends.

---

## 7. Files to change

| File | Change |
|---|---|
| `src/coastal_sst_data/processes/met.py` | **§0.** `net.retry` around the HRRR cycle, the ERA5 store open and `.load()`; gcsfs timeout; lock on `_ERA5_CACHE`; `_fetch_one` raises on failure instead of swallowing. |
| `src/coastal_sst_data/processes/met_overpass.py` | **§0.** Catch and tally the now-propagating fetch error at `:89`. |
| `tests/test_met.py` | **§0.** Rewrite `test_fetch_one_swallows_a_source_error` (`:216`) for the new split; add retry/timeout coverage. |
| `src/coastal_sst_data/scheduler.py` | **New.** `Task`, `run_graph`, gate semaphores. ~150 lines. |
| `src/coastal_sst_data/pipeline.py` | Build the `(product, aoi)` graph from `_modules_for` + `depends_on`; keep `jobs == 1` on today's path. Hoist `net.setup_gdal_env()`. |
| `src/coastal_sst_data/auth.py` | `login()` becomes a no-op when the recorded session is fresh (hazard 1). |
| `src/coastal_sst_data/config.py` | New `RuntimeSpec` block (`jobs`, `assemble_jobs`, `gates`) on `Project`. |
| `src/coastal_sst_data/cli.py` | `--jobs` / `--assemble-jobs` on `run`, `assemble`, `preprocess`. |
| `src/coastal_sst_data/processes/datacube.py`, `.../preprocess.py` | Additive `memory_budget_gb=` keyword on `assemble()`/`preprocess()`, threaded into `_build_eff`. |
| `src/coastal_sst_data/store.py` | Thread id in `_tag()` (hazard 8). |
| `src/coastal_sst_data/report.py` | Wall-clock span instead of summed elapsed (hazard 11). |
| logging | Thread-local `product/aoi` label filter (hazard 10). |

**No process module's `acquire()` signature or body changes.**

### Phasing

0. **Harden `met.py`** (§0). Independently valuable — it fixes a live silent-data-loss bug — and a
   prerequisite for the pool. Ships and verifies alone, before any concurrency exists.
1. **Prerequisites** — hazards 1, 4, 5, 7, 8, plus the logging label (10). No behaviour change; ship
   and verify green first.
2. **Scheduler + product×AoI acquisition**, gates, `--jobs`. The bulk of the win.
3. **Parallel assemble/preprocess** with the divided memory budget. Separate commit — this is the one
   that can OOM.
4. *(Optional)* Temporal sharding, §6.

---

## 8. Verification

### Step 0, verified on its own first

```bash
pytest tests/test_met.py tests/test_met_overpass.py tests/test_net.py
```

Assert three behaviours that do not hold today, using the existing fake-clock retry harness in
`tests/test_net.py` (no real sleeping, no network):

- a transient failure (503, timeout) from Herbie or the ERA5 `.load()` is **retried** with backoff
  and then succeeds;
- a failure that survives the retries **propagates** out of `_fetch_one` and is tallied by the caller
  with its real reason — *not* as `"no data"`;
- a genuine coverage miss (HRRR off-continent, empty ERA5 window) still returns `None` and still
  reports `"no data"`, unchanged.

Then a real single-AoI `met`-only run, confirming the run report distinguishes the two:

```bash
coastal-sst-data run --config config.yaml --products met --aoi <one> --jobs 1
```

### The rest

**The golden snapshot tests are a ready-made proof.** A correct parallel run must produce a
byte-identical cube:

```bash
pytest tests/test_datacube.py::test_golden_cube_is_unchanged
pytest tests/test_preprocess.py -k golden
```

(`tests/test_datacube.py:925`; regenerate only with `UPDATE_GOLDEN=1` — it must **not** be needed
here.)

**Ordering.** `tests/test_pipeline.py:17-50` already stubs every module's `acquire()` into a recorder
to assert dispatch order without network. Extend it to assert, at `jobs > 1`, that for **every** AoI:
`landsat` completes before `modis` starts, and all three sensors complete before `mur` and
`met_overpass` start — and that `jobs=1` still emits today's exact product-major sequence.

**New scheduler unit tests** — deps honoured, gate caps never exceeded (instrument the semaphore with
a peak counter), a failed task marks its dependents skipped rather than dropping them,
`KeyboardInterrupt` unwinds.

**End-to-end, on the real config, in this order:**

```bash
# 1. Unchanged serial baseline
coastal-sst-data run --config config.yaml --jobs 1 --dry-run

# 2. Parallel dry run — proves the graph and gates, touches no network payload
coastal-sst-data run --config config.yaml --jobs 8 --dry-run

# 3. One AoI, short date range, real network. Compare the run report against a serial run:
#    written/skipped/failed must match.
coastal-sst-data run --config config.yaml --aoi <one> --jobs 8 --assemble --preprocess

# 4. Integrity — must report a clean tree, no truncated files, no scratch leftovers
coastal-sst-data check --config config.yaml

# 5. Full run, watching RSS during assembly
/usr/bin/time -v coastal-sst-data run --config config.yaml --jobs 8 --assemble --preprocess
```

Step 4 is the load-bearing one: `store.scan` reads each file's **payload**, so it catches exactly the
truncation and scratch-collision class of bug that concurrent writes would introduce (`store.py:299`).

For assembly memory, confirm from the log line at `processes/datacube.py:1950` that each AoI reports
`budget ≈ total / assemble_jobs`, and that the existing >20% prediction-vs-`ru_maxrss` warning stays
quiet.
