"""coastal_sst_data -- acquire and grid coastal SST + covariate data products.

Public API for using the package as a library from another project:

    import coastal_sst_data as csd

    project = csd.load_config("config.yaml")   # validated Project
    grids = csd.project_grids(project)         # {aoi_name: AoiGrid}
    csd.run_pipeline(project, assemble=True)   # acquire everything + build cubes

The command-line interface (``coastal-sst-data``) is a thin wrapper over these
same functions -- see ``coastal_sst_data.cli``.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .config import DataProduct, Project, load_config, parse_config
from .grid import AoiGrid, compute_aoi_grid, project_grids
from .pipeline import run_pipeline

try:
    __version__ = _pkg_version("coastal_sst_data")
except PackageNotFoundError:              # running from a source tree, not installed
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "load_config",
    "parse_config",
    "Project",
    "DataProduct",
    "project_grids",
    "compute_aoi_grid",
    "AoiGrid",
    "run_pipeline",
]
