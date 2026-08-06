#!/usr/bin/env python3
"""
coastal_sst_data -- the derived-channel naming vocabulary shared by the preprocess steps.

The preprocess stage writes its output back into the ASSEMBLED cube (see `processes.preprocess`),
so a derived channel must never take the name of one the assembler wrote: `eco_sst_v002` has to
keep holding what the sensor delivered. Every step therefore emits under a distinguishing suffix,
and every step that SCANS the cube for its inputs (`<pre>_sst<ver>`, `cmems_*`, ...) has to skip
the derived names -- otherwise a second run treats last run's output as fresh input and produces
`eco_sst_v002_clean_clean`.

Both of those rules need the same list, and it must not drift between the two, so it lives here.
This module is deliberately dependency-free: `preprocess`, `cloud_filter` and `georef` all import
it, and it imports none of them.
"""

from __future__ import annotations

# ---- the suffixes each step writes under ---------------------------------------------- #
GAPFILLED = "_gapfilled"        # fill_water: the nearest-neighbour filled level-4 SST
FILL_FLAG = "_filled"           # fill_water: 1 where the value above was invented
CLEAN = "_clean"                # the cloud filters' screened product (sst / valid)
CORRECTED = "_georef_corrected"  # correct_georef: the raw fields on corrected geometry

# The corrected pass re-filters the corrected geometry into its own product, so the two suffixes
# stack. Spelled as a composition rather than a literal so the pair cannot drift apart.
CORRECTED_CLEAN = CORRECTED + CLEAN

# ---- what a cube scan must NOT mistake for a raw channel -------------------------------- #
# Scope: every suffix that can appear under a prefix a step scans (`<pre>_sst`, `<pre>_valid`,
# `cmems_`). The 1-D georef diagnostics (`<pre>_georef_dy`, ...) and `<pre>_scene_cloud_pct_<src>`
# are derived too, but no scan uses a prefix that reaches them, so listing them would be noise.
DERIVED_SUFFIXES: tuple[str, ...] = (
    GAPFILLED, FILL_FLAG,
    CLEAN, CORRECTED,                                  # CORRECTED_CLEAN ends with CLEAN
    "_cloudfiltered", "_metcloudfiltered", "_landcloudfiltered",
    "_water_elev", "_water_class",
)


def is_derived(name: str) -> bool:
    """True if `name` is a channel the preprocess stage produced, not one the assembler wrote."""
    return name.endswith(DERIVED_SUFFIXES)
